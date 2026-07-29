//! `CallSession` — the object a client drives for one call.
//!
//! ```text
//! start ──► push_local / push_remote (RTP threads, realtime-safe)
//!             │
//!             └─► pump() on a drain thread: resample to 16 kHz
//!   stop ──► seal both legs
//!   transcribe(&dyn Transcriber) ──► speaker-attributed, timed Transcript
//!   summarize(&dyn Summarizer)   ──► Summary (map-reduce for long calls)
//!   insight() ──► CallInsight for the UI
//! ```
//!
//! Two properties the design spec insists on and this type enforces:
//!
//! * **The captured audio is dropped the moment transcription finishes**, on
//!   every exit path including failure (§3.6 step 8). A `CallSession` never
//!   holds call audio after it has produced a transcript.
//! * **Nothing here panics.** Every state violation is an [`AiError`].
//!
//! Deferred to later phases, deliberately: the encrypted on-disk spill and
//! the `SQLite` `InsightStore` (§5), the serial background job queue (§3.6),
//! and consent/capture-mode policy (§5.5). This type is the unit those layers
//! will drive.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::audio::segment::{segment_call, slice_ms, SegmentationConfig};
use crate::audio::tap::{CallAudioTap, CapturedAudio, LegStats, TapConfig};
use crate::error::{AiError, Result};
use crate::llm::pipeline::{summarize, SummaryConfig};
use crate::llm::prompt::CallContext;
use crate::llm::Summarizer;
use crate::stt::{TranscribeRequest, Transcriber};
use crate::types::{
    CallInsight, InsightStatus, Speaker, Summary, Transcript, TranscriptSegment,
};
use crate::util::lock;

/// Language tag used when nothing could be detected.
const UNKNOWN_LANGUAGE: &str = "und";

/// How to run one call's capture and analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionConfig {
    /// Stable id shared with the call-history row (design spec §5.3).
    pub call_id: String,
    /// Forced recognition language, or `None` to auto-detect.
    pub language: Option<String>,
    pub tap: TapConfig,
    pub segmentation: SegmentationConfig,
}

impl SessionConfig {
    /// Defaults for a call with the given id.
    #[must_use]
    pub fn new(call_id: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            language: None,
            tap: TapConfig::default(),
            segmentation: SegmentationConfig::default(),
        }
    }

    /// Force a recognition language instead of auto-detecting.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

#[derive(Debug, Default)]
struct SessionInner {
    capture: Option<CapturedAudio>,
    transcript: Option<Transcript>,
    summary: Option<Summary>,
    stt_model_id: Option<String>,
    llm_model_id: Option<String>,
    status: Option<InsightStatus>,
}

/// One call's capture, transcription and summarization.
#[derive(Debug)]
pub struct CallSession {
    config: SessionConfig,
    tap: Arc<CallAudioTap>,
    created_at: i64,
    inner: Mutex<SessionInner>,
}

impl CallSession {
    /// Begin capturing.
    ///
    /// # Errors
    /// [`AiError::Config`] for an empty `call_id`.
    pub fn start(config: SessionConfig) -> Result<Arc<Self>> {
        if config.call_id.trim().is_empty() {
            return Err(AiError::Config("call_id must not be empty".into()));
        }
        let tap = Arc::new(CallAudioTap::new(&config.tap));
        Ok(Arc::new(Self {
            config,
            tap,
            created_at: unix_now(),
            inner: Mutex::new(SessionInner {
                status: Some(InsightStatus::Pending),
                ..SessionInner::default()
            }),
        }))
    }

    /// The call id this session belongs to.
    #[must_use]
    pub fn call_id(&self) -> &str {
        &self.config.call_id
    }

    /// Unix seconds at which capture began.
    #[must_use]
    pub const fn created_at(&self) -> i64 {
        self.created_at
    }

    /// The audio tap, for handing to the media layer.
    #[must_use]
    pub fn tap(&self) -> Arc<CallAudioTap> {
        Arc::clone(&self.tap)
    }

    /// Push a microphone frame. Realtime-safe; call from the RTP TX thread.
    ///
    /// # Errors
    /// As [`CallAudioTap::push_local`].
    pub fn push_local(&self, samples: &[i16], rate_hz: u32) -> Result<()> {
        self.tap.push_local(samples, rate_hz)
    }

    /// Push a decoded far-end frame. Realtime-safe; call from the RTP RX thread.
    ///
    /// # Errors
    /// As [`CallAudioTap::push_remote`].
    pub fn push_remote(&self, samples: &[i16], rate_hz: u32) -> Result<()> {
        self.tap.push_remote(samples, rate_hz)
    }

    /// Do the resampling work. Call from a drain thread, never from RTP.
    ///
    /// # Errors
    /// As [`CallAudioTap::pump`].
    pub fn pump(&self) -> Result<()> {
        self.tap.pump()
    }

    /// Run [`CallSession::pump`] on a background thread until the returned
    /// handle is dropped or [`DrainHandle::stop`] is called.
    #[must_use]
    pub fn spawn_drain_thread(self: &Arc<Self>, interval: Duration) -> DrainHandle {
        let stop = Arc::new(AtomicBool::new(false));
        let session = Arc::clone(self);
        let flag = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                if let Err(e) = session.pump() {
                    log::warn!("ai drain thread stopping: {e}");
                    break;
                }
                std::thread::sleep(interval);
            }
        });
        DrainHandle {
            stop,
            handle: Some(handle),
        }
    }

    /// Per-leg counters, safe to poll during the call.
    #[must_use]
    pub fn stats(&self) -> (LegStats, LegStats) {
        self.tap.stats()
    }

    /// Seal capture. Idempotent; safe to call from the hang-up path.
    ///
    /// # Errors
    /// As [`CallAudioTap::seal`].
    pub fn stop(&self) -> Result<()> {
        if self.tap.is_sealed() {
            return Ok(());
        }
        self.tap.seal()?;
        let capture = self.tap.take_capture()?;
        lock(&self.inner).capture = Some(capture);
        Ok(())
    }

    /// True once [`CallSession::stop`] has run.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.tap.is_sealed()
    }

    /// Abandon the capture and free the audio without transcribing it — the
    /// "discard this recording" affordance in the in-call indicator.
    pub fn discard(&self) {
        let _ = self.tap.seal();
        let _ = self.tap.take_capture();
        let mut inner = lock(&self.inner);
        inner.capture = None;
        inner.transcript = None;
        inner.summary = None;
        inner.status = Some(InsightStatus::Failed {
            reason: "discarded by the user".into(),
        });
    }

    /// Transcribe the captured audio, with speaker attribution and timings.
    ///
    /// **Consumes the audio.** On return — success or failure — this session
    /// no longer holds any call audio.
    ///
    /// # Errors
    /// - [`AiError::InvalidState`] if capture was never stopped, or the audio
    ///   has already been consumed.
    /// - [`AiError::Transcription`] if the engine failed on every segment.
    pub fn transcribe(&self, engine: &dyn Transcriber) -> Result<Transcript> {
        let capture = {
            let mut inner = lock(&self.inner);
            match inner.capture.take() {
                Some(c) => c,
                None => {
                    return inner.transcript.clone().ok_or_else(|| AiError::InvalidState {
                        expected: "stopped with audio captured".into(),
                        actual: if self.tap.is_sealed() {
                            "audio already consumed".into()
                        } else {
                            "still capturing".into()
                        },
                    })
                }
            }
        };

        self.set_status(InsightStatus::Transcribing);

        // The audio is dropped when `capture` goes out of scope at the end of
        // this function, on every path, including the `?` below.
        let result = self.run_transcription(&capture, engine);
        match result {
            Ok(transcript) => {
                let mut inner = lock(&self.inner);
                inner.stt_model_id = Some(engine.model_id());
                inner.transcript = Some(transcript.clone());
                inner.status = Some(InsightStatus::Complete);
                Ok(transcript)
            }
            Err(e) => {
                self.set_status(InsightStatus::Failed {
                    reason: e.to_string(),
                });
                Err(e)
            }
        }
    }

    fn run_transcription(
        &self,
        capture: &CapturedAudio,
        engine: &dyn Transcriber,
    ) -> Result<Transcript> {
        let segments = segment_call(
            &capture.local,
            &capture.remote,
            capture.rate_hz,
            &self.config.segmentation,
        );

        let mut out: Vec<TranscriptSegment> = Vec::new();
        let mut language: Option<String> = None;
        let mut failures = 0_usize;

        for seg in &segments {
            let leg = match seg.speaker {
                Speaker::Local => &capture.local,
                Speaker::Remote => &capture.remote,
            };
            let samples = slice_ms(leg, capture.rate_hz, seg.start_ms, seg.end_ms);
            if samples.is_empty() {
                continue;
            }

            let req = TranscribeRequest {
                samples,
                sample_rate: capture.rate_hz,
                speaker: seg.speaker,
                language: self.config.language.clone(),
            };
            let Ok(result) = engine.transcribe(&req) else {
                // One bad segment must not lose the whole call.
                failures += 1;
                continue;
            };

            if language.is_none() && !result.language.trim().is_empty() {
                language = Some(result.language.clone());
            }
            for u in result.utterances {
                let text = u.text.trim();
                if text.is_empty() {
                    continue;
                }
                out.push(TranscriptSegment {
                    speaker: seg.speaker,
                    start_ms: seg.start_ms.saturating_add(u.start_ms),
                    end_ms: seg
                        .start_ms
                        .saturating_add(u.end_ms)
                        .min(seg.end_ms.max(seg.start_ms)),
                    text: text.to_string(),
                    confidence: u.confidence.clamp(0.0, 1.0),
                });
            }
        }

        if failures > 0 && failures == segments.len() {
            return Err(AiError::Transcription(format!(
                "all {failures} segments failed to transcribe"
            )));
        }

        out.sort_by_key(|s| (s.start_ms, s.end_ms));

        Ok(Transcript {
            call_id: self.config.call_id.clone(),
            language: language
                .or_else(|| self.config.language.clone())
                .unwrap_or_else(|| UNKNOWN_LANGUAGE.to_string()),
            duration_ms: u32::try_from(capture.duration_ms).unwrap_or(u32::MAX),
            segments: out,
            narrowband: capture.narrowband,
            lost_audio: capture.lost_audio(),
        })
    }

    /// The transcript, once [`CallSession::transcribe`] has produced one.
    #[must_use]
    pub fn transcript(&self) -> Option<Transcript> {
        lock(&self.inner).transcript.clone()
    }

    /// Summarize the transcript.
    ///
    /// Single pass for short calls, map-reduce with overlap for long ones.
    ///
    /// # Errors
    /// - [`AiError::InvalidState`] if there is no transcript yet.
    /// - [`AiError::TranscriptTooShort`] — the call had too little speech;
    ///   the session status becomes [`InsightStatus::TranscriptOnly`].
    /// - Whatever [`crate::llm::summarize`] returns otherwise.
    pub fn summarize(
        &self,
        engine: &dyn Summarizer,
        ctx: &CallContext,
        cfg: &SummaryConfig,
    ) -> Result<Summary> {
        let transcript = self.transcript().ok_or_else(|| AiError::InvalidState {
            expected: "transcribed".into(),
            actual: "no transcript".into(),
        })?;

        self.set_status(InsightStatus::Summarizing);

        match summarize(&transcript, engine, ctx, cfg) {
            Ok(outcome) => {
                let mut inner = lock(&self.inner);
                inner.llm_model_id = Some(outcome.model_id);
                inner.summary = Some(outcome.summary.clone());
                inner.status = Some(InsightStatus::Complete);
                Ok(outcome.summary)
            }
            Err(e @ AiError::TranscriptTooShort { .. }) => {
                self.set_status(InsightStatus::TranscriptOnly {
                    reason: e.to_string(),
                });
                Err(e)
            }
            Err(e) => {
                self.set_status(InsightStatus::Failed {
                    reason: e.to_string(),
                });
                Err(e)
            }
        }
    }

    /// The summary, once one has been produced.
    #[must_use]
    pub fn summary(&self) -> Option<Summary> {
        lock(&self.inner).summary.clone()
    }

    /// Where this session is in its lifecycle.
    #[must_use]
    pub fn status(&self) -> InsightStatus {
        lock(&self.inner)
            .status
            .clone()
            .unwrap_or(InsightStatus::Pending)
    }

    /// Everything the Call Insights screen needs.
    #[must_use]
    pub fn insight(&self) -> CallInsight {
        let inner = lock(&self.inner);
        let duration_secs = inner
            .transcript
            .as_ref()
            .map_or(0, |t| t.duration_ms / 1000);
        CallInsight {
            call_id: self.config.call_id.clone(),
            created_at: self.created_at,
            duration_secs,
            transcript: inner.transcript.clone(),
            summary: inner.summary.clone(),
            stt_model_id: inner.stt_model_id.clone(),
            llm_model_id: inner.llm_model_id.clone(),
            status: inner
                .status
                .clone()
                .unwrap_or(InsightStatus::Pending),
        }
    }

    fn set_status(&self, status: InsightStatus) {
        lock(&self.inner).status = Some(status);
    }
}

/// Owns the background drain thread. Stops it on drop.
#[derive(Debug)]
pub struct DrainHandle {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl DrainHandle {
    /// Stop the thread and wait for it.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            // A panicking drain thread must not become a panic here.
            let _ = h.join();
        }
    }
}

impl Drop for DrainHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::llm::mock::MockSummarizer;
    use crate::stt::MockTranscriber;
    use crate::util::ms_to_samples;

    fn tone_i16(rate: u32, ms: u32, hz: f32, amp: f32) -> Vec<i16> {
        let n = ms_to_samples(u64::from(ms), rate);
        (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (amp * (2.0 * std::f32::consts::PI * hz * t).sin() * 32_767.0) as i16
            })
            .collect()
    }

    fn silence_i16(rate: u32, ms: u32) -> Vec<i16> {
        vec![0_i16; ms_to_samples(u64::from(ms), rate)]
    }

    /// Drive a PCMU call where the two legs take alternating turns.
    /// `turns` is a list of (speaker, `duration_ms`).
    fn run_call(session: &CallSession, turns: &[(Speaker, u32)]) {
        for (speaker, ms) in turns {
            let speech = tone_i16(8_000, *ms, 300.0, 0.6);
            let quiet = silence_i16(8_000, *ms);
            let (local, remote) = match speaker {
                Speaker::Local => (&speech, &quiet),
                Speaker::Remote => (&quiet, &speech),
            };
            for (lf, rf) in local.chunks(160).zip(remote.chunks(160)) {
                session.push_local(lf, 8_000).expect("push local");
                session.push_remote(rf, 8_000).expect("push remote");
            }
            session.pump().expect("pump");
        }
    }

    #[test]
    fn an_empty_call_id_is_rejected() {
        assert!(matches!(
            CallSession::start(SessionConfig::new("  ")),
            Err(AiError::Config(_))
        ));
    }

    #[test]
    fn a_call_produces_a_speaker_attributed_transcript_with_timings() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        run_call(
            &s,
            &[
                (Speaker::Remote, 1_200),
                (Speaker::Local, 1_200),
                (Speaker::Remote, 1_200),
            ],
        );
        s.stop().expect("stop");

        let stt = MockTranscriber::scripted([
            "Hi, it's Sarah",
            "Hello Sarah",
            "Can we move the review",
        ]);
        let t = s.transcribe(&stt).expect("transcribe");

        assert_eq!(t.call_id, "call-1");
        assert_eq!(t.language, "en");
        assert!(t.narrowband, "8 kHz PCMU must be flagged narrowband");
        assert!(!t.lost_audio);
        assert_eq!(t.segments.len(), 3, "{:?}", t.segments);

        assert_eq!(
            t.segments.iter().map(|s| s.speaker).collect::<Vec<_>>(),
            vec![Speaker::Remote, Speaker::Local, Speaker::Remote]
        );
        assert_eq!(t.segments[0].text, "Hi, it's Sarah");
        assert_eq!(t.segments[2].text, "Can we move the review");

        // Timings are absolute in the call, ordered, and inside the call.
        assert!(t.segments.windows(2).all(|w| w[0].start_ms <= w[1].start_ms));
        assert!(t.segments[0].start_ms < 300, "{}", t.segments[0].start_ms);
        assert!(t.segments[1].start_ms >= 1_000, "{}", t.segments[1].start_ms);
        assert!(t.segments.iter().all(|s| s.end_ms <= t.duration_ms + 500));
    }

    #[test]
    fn transcribing_frees_the_captured_audio() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        run_call(&s, &[(Speaker::Local, 800)]);
        s.stop().expect("stop");

        assert!(lock(&s.inner).capture.is_some());
        s.transcribe(&MockTranscriber::new()).expect("transcribe");
        assert!(
            lock(&s.inner).capture.is_none(),
            "call audio must not outlive transcription"
        );
    }

    #[test]
    fn transcribing_twice_returns_the_stored_transcript_without_audio() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        run_call(&s, &[(Speaker::Local, 800)]);
        s.stop().expect("stop");

        let stt = MockTranscriber::new();
        let first = s.transcribe(&stt).expect("first");
        let calls = stt.calls();
        let second = s.transcribe(&stt).expect("second");
        assert_eq!(first, second);
        assert_eq!(stt.calls(), calls, "must not re-run the engine");
    }

    #[test]
    fn transcribing_before_stopping_is_a_state_error() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        run_call(&s, &[(Speaker::Local, 400)]);
        match s.transcribe(&MockTranscriber::new()) {
            Err(AiError::InvalidState { actual, .. }) => assert_eq!(actual, "still capturing"),
            other => panic!("expected InvalidState, got {other:?}"),
        }
    }

    #[test]
    fn stop_is_idempotent() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        run_call(&s, &[(Speaker::Local, 400)]);
        s.stop().expect("stop");
        s.stop().expect("stop again");
        assert!(s.is_stopped());
    }

    #[test]
    fn a_silent_call_transcribes_to_nothing_rather_than_hallucinating() {
        let s = CallSession::start(SessionConfig::new("quiet")).expect("start");
        for _ in 0..50 {
            s.push_local(&silence_i16(8_000, 20), 8_000).expect("push");
            s.push_remote(&silence_i16(8_000, 20), 8_000).expect("push");
        }
        s.pump().expect("pump");
        s.stop().expect("stop");

        let t = s.transcribe(&MockTranscriber::new()).expect("transcribe");
        assert!(t.segments.is_empty());
        assert!(t.is_empty());
    }

    #[test]
    fn a_failing_engine_leaves_the_session_failed_not_panicking() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        run_call(&s, &[(Speaker::Local, 900)]);
        s.stop().expect("stop");

        let err = s
            .transcribe(&MockTranscriber::failing("no model"))
            .expect_err("must fail");
        assert!(matches!(err, AiError::Transcription(_)));
        assert!(matches!(s.status(), InsightStatus::Failed { .. }));
        assert!(s.transcript().is_none());
        assert!(
            lock(&s.inner).capture.is_none(),
            "audio must be freed even on failure"
        );
    }

    #[test]
    fn discarding_a_call_frees_everything() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        run_call(&s, &[(Speaker::Local, 900)]);
        s.discard();
        assert!(lock(&s.inner).capture.is_none());
        assert!(matches!(s.status(), InsightStatus::Failed { .. }));
    }

    #[test]
    fn the_status_walks_the_expected_lifecycle() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        assert_eq!(s.status(), InsightStatus::Pending);

        // Enough speech to clear the 30 s summarization floor.
        run_call(
            &s,
            &[
                (Speaker::Remote, 12_000),
                (Speaker::Local, 12_000),
                (Speaker::Remote, 12_000),
            ],
        );
        s.stop().expect("stop");

        s.transcribe(&MockTranscriber::new()).expect("transcribe");
        assert_eq!(s.status(), InsightStatus::Complete);

        s.summarize(
            &MockSummarizer::new(),
            &CallContext::default(),
            &SummaryConfig::default(),
        )
        .expect("summary");
        assert_eq!(s.status(), InsightStatus::Complete);

        let insight = s.insight();
        assert_eq!(insight.call_id, "call-1");
        assert!(insight.transcript.is_some());
        assert!(insight.summary.is_some());
        assert_eq!(insight.stt_model_id.as_deref(), Some("mock-stt"));
        assert_eq!(insight.llm_model_id.as_deref(), Some("mock-llm"));
        assert!(insight.duration_secs >= 30, "{}", insight.duration_secs);
        assert!(insight.created_at > 0);
    }

    #[test]
    fn a_short_call_is_marked_transcript_only_not_failed() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        run_call(&s, &[(Speaker::Local, 1_500)]);
        s.stop().expect("stop");
        s.transcribe(&MockTranscriber::new()).expect("transcribe");

        let err = s
            .summarize(
                &MockSummarizer::new(),
                &CallContext::default(),
                &SummaryConfig::default(),
            )
            .expect_err("too short");
        assert!(matches!(err, AiError::TranscriptTooShort { .. }));
        assert!(matches!(s.status(), InsightStatus::TranscriptOnly { .. }));

        // The transcript is still there — that is the whole point.
        let insight = s.insight();
        assert!(insight.transcript.is_some());
        assert!(insight.summary.is_none());
    }

    #[test]
    fn summarizing_without_a_transcript_is_a_state_error() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        assert!(matches!(
            s.summarize(
                &MockSummarizer::new(),
                &CallContext::default(),
                &SummaryConfig::default()
            ),
            Err(AiError::InvalidState { .. })
        ));
    }

    #[test]
    fn a_forced_language_is_carried_through_to_the_transcript() {
        let s = CallSession::start(SessionConfig::new("call-1").with_language("de"))
            .expect("start");
        run_call(&s, &[(Speaker::Local, 900)]);
        s.stop().expect("stop");
        let t = s.transcribe(&MockTranscriber::new()).expect("transcribe");
        assert_eq!(t.language, "de");
    }

    #[test]
    fn opus_at_48k_is_not_flagged_narrowband() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        let speech = tone_i16(48_000, 900, 300.0, 0.6);
        for frame in speech.chunks(960) {
            s.push_remote(frame, 48_000).expect("push");
        }
        s.pump().expect("pump");
        s.stop().expect("stop");
        let t = s.transcribe(&MockTranscriber::new()).expect("transcribe");
        assert!(!t.narrowband);
    }

    #[test]
    fn the_drain_thread_keeps_the_ring_from_overflowing() {
        let cfg = SessionConfig {
            tap: TapConfig {
                ring_ms: 200,
                ..TapConfig::default()
            },
            ..SessionConfig::new("call-1")
        };
        let s = CallSession::start(cfg).expect("start");
        let drain = s.spawn_drain_thread(Duration::from_millis(5));

        let speech = tone_i16(8_000, 1_000, 300.0, 0.6);
        for frame in speech.chunks(160) {
            s.push_local(frame, 8_000).expect("push");
            std::thread::sleep(Duration::from_millis(1));
        }
        drain.stop();

        s.stop().expect("stop");
        let t = s.transcribe(&MockTranscriber::new()).expect("transcribe");
        assert!(!t.lost_audio, "the drain thread should have kept up");
    }

    #[test]
    fn a_dropped_drain_handle_stops_the_thread() {
        let s = CallSession::start(SessionConfig::new("call-1")).expect("start");
        {
            let _drain = s.spawn_drain_thread(Duration::from_millis(1));
            s.push_local(&silence_i16(8_000, 1_000), 8_000).expect("push");
            std::thread::sleep(Duration::from_millis(200));
        }

        // The thread ran while the handle was alive...
        let (before, _) = s.stats();
        assert!(before.captured_ms > 0, "the drain thread never pumped");

        // ...and `Drop` stopped and joined it, so nothing drains the ring now.
        s.push_local(&silence_i16(8_000, 1_000), 8_000).expect("push");
        std::thread::sleep(Duration::from_millis(200));
        let (after, _) = s.stats();
        assert_eq!(
            after.captured_ms, before.captured_ms,
            "the drain thread outlived the handle that owns it"
        );

        s.stop().expect("stop");
    }

    #[test]
    fn a_long_call_summary_uses_map_reduce_end_to_end() {
        let s = CallSession::start(SessionConfig::new("long-call")).expect("start");
        // 12 turns x 12 s of speech, transcribed as long scripted lines, so
        // the transcript comfortably exceeds a 1k context.
        let turns: Vec<(Speaker, u32)> = (0..12)
            .map(|i| {
                (
                    if i % 2 == 0 {
                        Speaker::Remote
                    } else {
                        Speaker::Local
                    },
                    12_000,
                )
            })
            .collect();
        run_call(&s, &turns);
        s.stop().expect("stop");

        let line = vec!["word"; 120].join(" ");
        let stt = MockTranscriber::scripted(std::iter::repeat_n(line, 64));
        let t = s.transcribe(&stt).expect("transcribe");
        assert!(t.segments.len() >= 12);

        let llm = MockSummarizer::new().with_context(1_024);
        let summary = s
            .summarize(&llm, &CallContext::default(), &SummaryConfig::default())
            .expect("summary");

        assert!(llm.calls() > 2, "expected map-reduce, got {} calls", llm.calls());
        assert!(!summary.degraded);
        assert!(!summary.headline.is_empty());
    }
}

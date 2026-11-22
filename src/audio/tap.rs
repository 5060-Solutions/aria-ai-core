//! `CallAudioTap` — the object the media layer pushes frames into.
//!
//! Contract, taken from `aria_mobile_core::media::PlatformAudioBridge`:
//! mono `i16` PCM, 8 kHz for PCMU/PCMA/G.729 or 48 kHz for Opus, arriving in
//! 20 ms frames, from two independent legs.
//!
//! Split of responsibilities, per design spec §3.4:
//!
//! * [`CallAudioTap::push_local`] / [`CallAudioTap::push_remote`] run on the
//!   RTP threads. They do a bounded ring write and nothing else — no
//!   allocation beyond the one-time ring, no resampling, no I/O, no blocking.
//! * [`CallAudioTap::pump`] runs on the host's drain thread and does the
//!   expensive work: resample to 16 kHz mono `f32` and accumulate.
//!
//! Phase 1 accumulates in memory with a hard cap. The encrypted on-disk spill
//! the spec describes (§3.4/§5.2) is Phase 2, and slots in behind `pump`.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

use crate::audio::resample::{SincResampler, TARGET_RATE_HZ};
use crate::audio::ring::{PcmRing, RingStats};
use crate::error::{AiError, Result};
use crate::util::{lock, ms_to_samples, samples_to_ms};

/// Ring capacity per leg. Two seconds is far more slack than a 20 ms RTP grid
/// needs, and costs 192 KB at the worst-case 48 kHz.
const DEFAULT_RING_MS: u64 = 2_000;

/// Rates the platform bridges actually produce.
const SUPPORTED_RATES: [u32; 4] = [8_000, 16_000, 32_000, 48_000];

/// Tunables for [`CallAudioTap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct TapConfig {
    /// Per-leg ring buffer depth in milliseconds.
    pub ring_ms: u64,
    /// Hard cap on captured audio per leg (design spec §3.4 mitigation 3).
    pub max_capture_secs: u32,
}

impl Default for TapConfig {
    fn default() -> Self {
        Self {
            ring_ms: DEFAULT_RING_MS,
            // 120 minutes, the spec's default.
            max_capture_secs: 120 * 60,
        }
    }
}

/// What a leg looked like after the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct LegStats {
    pub ring: RingStats,
    /// Source rate last seen from the media layer, 0 if the leg never ran.
    pub source_rate_hz: u32,
    /// Milliseconds of 16 kHz audio accumulated.
    pub captured_ms: u64,
}

/// Both legs of one call, buffered and resampled to 16 kHz mono `f32`.
#[derive(Debug)]
pub struct CallAudioTap {
    local: Leg,
    remote: Leg,
    max_samples: usize,
    truncated: AtomicBool,
    sealed: AtomicBool,
}

#[derive(Debug)]
struct Leg {
    ring: PcmRing,
    source_rate: AtomicU32,
    state: Mutex<LegState>,
}

#[derive(Debug, Default)]
struct LegState {
    resampler: Option<SincResampler>,
    out: Vec<f32>,
}

impl Leg {
    fn new(ring_ms: u64) -> Self {
        // Size for the worst-case source rate so a mid-call codec change
        // cannot silently shrink the buffer.
        let cap = ms_to_samples(ring_ms, 48_000);
        Self {
            ring: PcmRing::with_capacity(cap),
            source_rate: AtomicU32::new(0),
            state: Mutex::new(LegState::default()),
        }
    }

    fn push(&self, samples: &[i16], rate_hz: u32) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        if !SUPPORTED_RATES.contains(&rate_hz) {
            return Err(AiError::UnsupportedSampleRate { rate: rate_hz });
        }
        self.source_rate.store(rate_hz, Ordering::Relaxed);
        self.ring.push(samples);
        Ok(())
    }

    /// Drain the ring, resample, accumulate. Returns samples appended.
    fn pump(&self, max_samples: usize) -> Result<usize> {
        let rate = self.source_rate.load(Ordering::Relaxed);
        if rate == 0 {
            return Ok(0);
        }
        let mut scratch = Vec::new();
        let drained = self.ring.drain_all(&mut scratch);
        if drained == 0 {
            return Ok(0);
        }

        let mut st = lock(&self.state);
        let needs_new = st
            .resampler
            .as_ref()
            .is_none_or(|r| r.input_rate() != rate);
        if needs_new {
            // Codec renegotiated mid-call: flush the old filter, start a new one.
            if let Some(old) = st.resampler.as_mut() {
                let mut tail = Vec::new();
                old.flush(&mut tail);
                append_capped(&mut st.out, &tail, max_samples);
            }
            st.resampler = Some(SincResampler::to_target(rate)?);
        }

        let mut produced = Vec::new();
        if let Some(r) = st.resampler.as_mut() {
            r.process_i16(&scratch, &mut produced);
        }
        Ok(append_capped(&mut st.out, &produced, max_samples))
    }

    fn seal(&self, max_samples: usize) {
        let mut st = lock(&self.state);
        let mut tail = Vec::new();
        if let Some(r) = st.resampler.as_mut() {
            r.flush(&mut tail);
        }
        append_capped(&mut st.out, &tail, max_samples);
        st.resampler = None;
    }

    fn take(&self) -> Vec<f32> {
        std::mem::take(&mut lock(&self.state).out)
    }

    fn stats(&self) -> LegStats {
        let captured = lock(&self.state).out.len();
        LegStats {
            ring: self.ring.stats(),
            source_rate_hz: self.source_rate.load(Ordering::Relaxed),
            captured_ms: samples_to_ms(captured, TARGET_RATE_HZ),
        }
    }
}

/// Append up to the cap; returns how many were actually appended.
fn append_capped(out: &mut Vec<f32>, src: &[f32], max_samples: usize) -> usize {
    if out.len() >= max_samples {
        return 0;
    }
    let room = max_samples - out.len();
    let n = src.len().min(room);
    out.extend_from_slice(src.get(..n).unwrap_or(&[]));
    n
}

impl CallAudioTap {
    /// Create a tap. Allocates both rings up front so the RTP threads never
    /// allocate.
    #[must_use]
    pub fn new(cfg: &TapConfig) -> Self {
        let ring_ms = cfg.ring_ms.clamp(100, 30_000);
        let max_samples =
            ms_to_samples(u64::from(cfg.max_capture_secs).saturating_mul(1_000), TARGET_RATE_HZ)
                .max(TARGET_RATE_HZ as usize);
        Self {
            local: Leg::new(ring_ms),
            remote: Leg::new(ring_ms),
            max_samples,
            truncated: AtomicBool::new(false),
            sealed: AtomicBool::new(false),
        }
    }

    /// Push a frame of microphone audio. Realtime-safe.
    ///
    /// # Errors
    /// [`AiError::UnsupportedSampleRate`] for a rate no codec produces.
    /// Rejects pushes after [`CallAudioTap::seal`].
    pub fn push_local(&self, samples: &[i16], rate_hz: u32) -> Result<()> {
        self.check_open()?;
        self.local.push(samples, rate_hz)
    }

    /// Push a frame of decoded far-end audio. Realtime-safe.
    ///
    /// # Errors
    /// As [`CallAudioTap::push_local`].
    pub fn push_remote(&self, samples: &[i16], rate_hz: u32) -> Result<()> {
        self.check_open()?;
        self.remote.push(samples, rate_hz)
    }

    /// Drain both rings and resample. Call from the host's drain thread every
    /// few hundred milliseconds; never from an RTP thread.
    ///
    /// # Errors
    /// [`AiError::UnsupportedSampleRate`] if a leg reported an impossible rate.
    pub fn pump(&self) -> Result<()> {
        let l = self.local.pump(self.max_samples)?;
        let r = self.remote.pump(self.max_samples)?;
        if (l == 0 || r == 0) && self.at_capacity() {
            self.truncated.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Final pump + filter flush. Idempotent.
    ///
    /// # Errors
    /// As [`CallAudioTap::pump`].
    pub fn seal(&self) -> Result<()> {
        if self.sealed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        // One last drain before the flag stops further pushes.
        self.local.pump(self.max_samples)?;
        self.remote.pump(self.max_samples)?;
        self.local.seal(self.max_samples);
        self.remote.seal(self.max_samples);
        Ok(())
    }

    /// True once [`CallAudioTap::seal`] has run.
    #[must_use]
    pub fn is_sealed(&self) -> bool {
        self.sealed.load(Ordering::SeqCst)
    }

    /// True if capture hit `max_capture_secs` and audio was discarded.
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        self.truncated.load(Ordering::Relaxed)
    }

    /// Per-leg counters, safe to poll during a call.
    #[must_use]
    pub fn stats(&self) -> (LegStats, LegStats) {
        (self.local.stats(), self.remote.stats())
    }

    /// Take the captured 16 kHz audio, leaving the tap empty.
    ///
    /// # Errors
    /// [`AiError::InvalidState`] if the tap has not been sealed.
    pub fn take_capture(&self) -> Result<CapturedAudio> {
        if !self.is_sealed() {
            return Err(AiError::InvalidState {
                expected: "sealed".into(),
                actual: "capturing".into(),
            });
        }
        let (local_stats, remote_stats) = self.stats();
        let local = self.local.take();
        let remote = self.remote.take();
        let duration_ms = samples_to_ms(local.len().max(remote.len()), TARGET_RATE_HZ);
        let narrowband = [local_stats.source_rate_hz, remote_stats.source_rate_hz]
            .iter()
            .any(|r| *r != 0 && *r <= 8_000);
        Ok(CapturedAudio {
            local,
            remote,
            rate_hz: TARGET_RATE_HZ,
            duration_ms,
            truncated: self.is_truncated(),
            narrowband,
            local_stats,
            remote_stats,
        })
    }

    fn check_open(&self) -> Result<()> {
        if self.is_sealed() {
            return Err(AiError::InvalidState {
                expected: "capturing".into(),
                actual: "sealed".into(),
            });
        }
        Ok(())
    }

    fn at_capacity(&self) -> bool {
        lock(&self.local.state).out.len() >= self.max_samples
            || lock(&self.remote.state).out.len() >= self.max_samples
    }
}

/// The sealed, resampled result of one call's capture.
#[derive(Debug, Clone)]
pub struct CapturedAudio {
    /// Microphone leg, 16 kHz mono `f32`.
    pub local: Vec<f32>,
    /// Far-end leg, 16 kHz mono `f32`.
    pub remote: Vec<f32>,
    pub rate_hz: u32,
    pub duration_ms: u64,
    /// Capture stopped early at `max_capture_secs`.
    pub truncated: bool,
    /// At least one leg was carried at 8 kHz — accuracy warning territory.
    pub narrowband: bool,
    pub local_stats: LegStats,
    pub remote_stats: LegStats,
}

impl CapturedAudio {
    /// True when either leg lost samples to ring overflow.
    #[must_use]
    pub const fn lost_audio(&self) -> bool {
        self.local_stats.ring.lost_audio() || self.remote_stats.ring.lost_audio()
    }

    /// True when neither leg captured anything.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.local.is_empty() && self.remote.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    fn tone_i16(rate: u32, ms: u32, hz: f32, amp: f32) -> Vec<i16> {
        let n = ms_to_samples(u64::from(ms), rate);
        (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (amp * (2.0 * std::f32::consts::PI * hz * t).sin() * 32_767.0) as i16
            })
            .collect()
    }

    #[test]
    fn pcmu_frames_become_16k_audio_at_the_right_length() {
        let tap = CallAudioTap::new(&TapConfig::default());
        let audio = tone_i16(8_000, 1_000, 440.0, 0.5);
        for frame in audio.chunks(160) {
            tap.push_local(frame, 8_000).expect("push");
            tap.pump().expect("pump");
        }
        tap.seal().expect("seal");
        let cap = tap.take_capture().expect("capture");

        assert_eq!(cap.rate_hz, 16_000);
        assert!(cap.local.len().abs_diff(16_000) <= 8, "{}", cap.local.len());
        assert_eq!(cap.remote, [] as [f32; 0]);
        assert!(cap.narrowband);
        assert!(!cap.truncated);
        assert!(!cap.lost_audio());
    }

    #[test]
    fn opus_48k_frames_decimate_to_16k() {
        let tap = CallAudioTap::new(&TapConfig::default());
        let audio = tone_i16(48_000, 1_000, 440.0, 0.5);
        for frame in audio.chunks(960) {
            tap.push_remote(frame, 48_000).expect("push");
        }
        tap.seal().expect("seal");
        let cap = tap.take_capture().expect("capture");
        assert!(cap.remote.len().abs_diff(16_000) <= 8, "{}", cap.remote.len());
        assert!(!cap.narrowband, "48k Opus must not be flagged narrowband");
    }

    #[test]
    fn both_legs_stay_time_aligned() {
        let tap = CallAudioTap::new(&TapConfig::default());
        // Local speaks for the first 500 ms, remote for the second 500 ms;
        // both legs get a full second of frames (silence substituted), which
        // is exactly what media.rs does on mute/underrun.
        let speech = tone_i16(8_000, 500, 300.0, 0.6);
        let quiet = vec![0_i16; ms_to_samples(500, 8_000)];

        let local: Vec<i16> = speech.iter().chain(quiet.iter()).copied().collect();
        let remote: Vec<i16> = quiet.iter().chain(speech.iter()).copied().collect();

        for (lf, rf) in local.chunks(160).zip(remote.chunks(160)) {
            tap.push_local(lf, 8_000).expect("push");
            tap.push_remote(rf, 8_000).expect("push");
            tap.pump().expect("pump");
        }
        tap.seal().expect("seal");
        let cap = tap.take_capture().expect("capture");

        assert_eq!(cap.local.len(), cap.remote.len());
        let segs = crate::audio::segment::segment_call(
            &cap.local,
            &cap.remote,
            cap.rate_hz,
            &crate::audio::segment::SegmentationConfig::default(),
        );
        assert_eq!(segs.len(), 2, "{segs:?}");
        assert_eq!(segs[0].speaker, crate::types::Speaker::Local);
        assert_eq!(segs[1].speaker, crate::types::Speaker::Remote);
        assert!(segs[1].start_ms >= 400, "remote turn at {}", segs[1].start_ms);
    }

    #[test]
    fn unsupported_rate_is_an_error_not_a_panic() {
        let tap = CallAudioTap::new(&TapConfig::default());
        assert!(matches!(
            tap.push_local(&[0, 1, 2], 11_025),
            Err(AiError::UnsupportedSampleRate { rate: 11_025 })
        ));
    }

    #[test]
    fn pushing_after_seal_is_rejected() {
        let tap = CallAudioTap::new(&TapConfig::default());
        tap.push_local(&[1; 160], 8_000).expect("push");
        tap.seal().expect("seal");
        assert!(matches!(
            tap.push_local(&[1; 160], 8_000),
            Err(AiError::InvalidState { .. })
        ));
        // Seal is idempotent.
        assert!(tap.seal().is_ok());
    }

    #[test]
    fn taking_capture_before_seal_is_rejected() {
        let tap = CallAudioTap::new(&TapConfig::default());
        assert!(matches!(
            tap.take_capture(),
            Err(AiError::InvalidState { .. })
        ));
    }

    #[test]
    fn capture_is_capped_and_flagged_as_truncated() {
        let tap = CallAudioTap::new(&TapConfig {
            ring_ms: 500,
            max_capture_secs: 1,
        });
        // Push 3 seconds of 8 kHz through a 1-second cap.
        let audio = tone_i16(8_000, 3_000, 300.0, 0.5);
        for frame in audio.chunks(160) {
            let _ = tap.push_local(frame, 8_000);
            let _ = tap.pump();
        }
        let _ = tap.pump();
        tap.seal().expect("seal");
        let cap = tap.take_capture().expect("capture");
        assert_eq!(cap.local.len(), 16_000);
        assert!(cap.truncated);
    }

    #[test]
    fn a_stalled_drain_thread_drops_oldest_and_reports_it() {
        // Rings are sized for the worst-case 48 kHz source, so 200 ms of ring
        // holds 9,600 samples: 2 s of 8 kHz audio (16,000) overruns it.
        let tap = CallAudioTap::new(&TapConfig {
            ring_ms: 200,
            ..TapConfig::default()
        });
        let audio = tone_i16(8_000, 2_000, 300.0, 0.5);
        for frame in audio.chunks(160) {
            tap.push_local(frame, 8_000).expect("push");
        }
        tap.seal().expect("seal");
        let cap = tap.take_capture().expect("capture");
        assert!(cap.lost_audio());
        assert!(cap.local_stats.ring.dropped_samples > 0);
        // What survived is the tail: ring depth at 48 kHz sizing, so all of
        // the last 200 ms at least.
        assert_ne!(cap.local, [] as [f32; 0]);
    }

    #[test]
    fn a_mid_call_codec_change_does_not_lose_the_stream() {
        let tap = CallAudioTap::new(&TapConfig::default());
        for frame in tone_i16(8_000, 400, 300.0, 0.5).chunks(160) {
            tap.push_local(frame, 8_000).expect("push");
            tap.pump().expect("pump");
        }
        for frame in tone_i16(48_000, 400, 300.0, 0.5).chunks(960) {
            tap.push_local(frame, 48_000).expect("push");
            tap.pump().expect("pump");
        }
        tap.seal().expect("seal");
        let cap = tap.take_capture().expect("capture");
        // ~800 ms of 16 kHz audio.
        assert!(cap.local.len().abs_diff(12_800) <= 64, "{}", cap.local.len());
        assert_eq!(cap.local_stats.source_rate_hz, 48_000);
    }

    #[test]
    fn a_call_with_no_audio_yields_an_empty_capture() {
        let tap = CallAudioTap::new(&TapConfig::default());
        tap.seal().expect("seal");
        let cap = tap.take_capture().expect("capture");
        assert!(cap.is_empty());
        assert_eq!(cap.duration_ms, 0);
        assert!(!cap.narrowband);
    }
}

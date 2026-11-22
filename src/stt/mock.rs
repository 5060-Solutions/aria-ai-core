//! A deterministic [`Transcriber`] for tests and for developing the layers
//! above inference before any weights exist.
//!
//! Deliberately mimics three behaviours of the real thing that the pipeline
//! has to cope with: silence produces *no* output (the anti-hallucination
//! property we are choosing Parakeet for), identical audio produces identical
//! text, and failures come back as errors rather than panics.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::audio::vad::rms;
use crate::error::{AiError, Result};
use crate::stt::{TranscribeOutput, TranscribeRequest, Transcriber, Utterance};
use crate::util::{lock, samples_to_ms};

/// Below this RMS the mock returns nothing, like a well-behaved engine.
const SILENCE_RMS: f32 = 0.005;

/// Deterministic stand-in for a real STT engine.
#[derive(Debug)]
pub struct MockTranscriber {
    model_id: String,
    language: String,
    script: Mutex<VecDeque<String>>,
    fail_with: Option<String>,
    calls: AtomicUsize,
}

impl Default for MockTranscriber {
    fn default() -> Self {
        Self::new()
    }
}

impl MockTranscriber {
    /// A mock that synthesises text from the audio it is given.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_id: "mock-stt".into(),
            language: "en".into(),
            script: Mutex::new(VecDeque::new()),
            fail_with: None,
            calls: AtomicUsize::new(0),
        }
    }

    /// A mock that returns `lines` in order, one per non-silent call, then
    /// falls back to synthesised text.
    #[must_use]
    pub fn scripted<I, S>(lines: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut m = Self::new();
        m.script = Mutex::new(lines.into_iter().map(Into::into).collect());
        m
    }

    /// A mock that always fails, for exercising error paths.
    #[must_use]
    pub fn failing(reason: impl Into<String>) -> Self {
        Self {
            fail_with: Some(reason.into()),
            ..Self::new()
        }
    }

    /// Override the reported model id.
    #[must_use]
    pub fn with_model_id(mut self, id: impl Into<String>) -> Self {
        self.model_id = id.into();
        self
    }

    /// Override the reported language.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// How many times [`Transcriber::transcribe`] has been called.
    #[must_use]
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Transcriber for MockTranscriber {
    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn transcribe(&self, req: &TranscribeRequest<'_>) -> Result<TranscribeOutput> {
        self.calls.fetch_add(1, Ordering::SeqCst);

        if let Some(reason) = &self.fail_with {
            return Err(AiError::Transcription(reason.clone()));
        }
        if req.sample_rate == 0 {
            return Err(AiError::InvalidAudio("sample rate is zero".into()));
        }

        let language = req
            .language
            .clone()
            .unwrap_or_else(|| self.language.clone());

        if req.samples.is_empty() || rms(req.samples) < SILENCE_RMS {
            return Ok(TranscribeOutput::empty(language));
        }

        let duration_ms =
            u32::try_from(samples_to_ms(req.samples.len(), req.sample_rate)).unwrap_or(u32::MAX);

        let text = lock(&self.script)
            .pop_front()
            .unwrap_or_else(|| synthetic_text(req, duration_ms));

        Ok(TranscribeOutput {
            utterances: vec![Utterance {
                start_ms: 0,
                end_ms: duration_ms,
                text,
                confidence: 0.9,
            }],
            language,
        })
    }
}

/// Stable pseudo-text: the same audio always yields the same words.
fn synthetic_text(req: &TranscribeRequest<'_>, duration_ms: u32) -> String {
    const WORDS: [&str; 8] = [
        "hello", "yes", "the", "meeting", "tomorrow", "confirm", "budget", "thanks",
    ];
    // FNV-1a over a decimated view of the samples — cheap, deterministic and
    // independent of buffer length beyond the content itself.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for s in req.samples.iter().step_by(97) {
        for b in s.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    // Roughly 2.5 words per second, at least one.
    let count = ((duration_ms / 400).max(1) as usize).min(24);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let idx = ((hash >> ((i % 8) * 8)) as usize) % WORDS.len();
        out.push(WORDS[idx]);
    }
    format!(
        "{} {}",
        req.speaker.prompt_label().to_lowercase(),
        out.join(" ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Speaker;

    fn tone(ms: u32, amp: f32) -> Vec<f32> {
        let n = crate::util::ms_to_samples(u64::from(ms), 16_000);
        #[allow(clippy::cast_precision_loss)]
        (0..n).map(|i| amp * (i as f32 * 0.05).sin()).collect()
    }

    fn req(samples: &[f32], speaker: Speaker) -> TranscribeRequest<'_> {
        TranscribeRequest {
            samples,
            sample_rate: 16_000,
            speaker,
            language: None,
        }
    }

    #[test]
    fn silence_produces_no_utterances() {
        let m = MockTranscriber::new();
        let quiet = vec![0.0_f32; 16_000];
        let out = m.transcribe(&req(&quiet, Speaker::Local)).expect("ok");
        assert_eq!(out.utterances, []);
        assert_eq!(out.language, "en");
        assert_eq!(m.calls(), 1);
    }

    #[test]
    fn empty_audio_is_not_an_error() {
        let m = MockTranscriber::new();
        let out = m.transcribe(&req(&[], Speaker::Remote)).expect("ok");
        assert_eq!(out.utterances, []);
    }

    #[test]
    fn identical_audio_yields_identical_text() {
        let m = MockTranscriber::new();
        let audio = tone(2_000, 0.5);
        let a = m.transcribe(&req(&audio, Speaker::Local)).expect("ok");
        let b = m.transcribe(&req(&audio, Speaker::Local)).expect("ok");
        assert_eq!(a, b);
        assert_ne!(a.utterances, []);
    }

    #[test]
    fn timings_are_relative_to_the_buffer() {
        let m = MockTranscriber::new();
        let audio = tone(1_500, 0.5);
        let out = m.transcribe(&req(&audio, Speaker::Local)).expect("ok");
        let u = out.utterances.first().expect("one utterance");
        assert_eq!(u.start_ms, 0);
        assert_eq!(u.end_ms, 1_500);
    }

    #[test]
    fn a_script_is_consumed_in_order_then_falls_back() {
        let m = MockTranscriber::scripted(["first line", "second line"]);
        let audio = tone(1_000, 0.5);
        assert_eq!(
            m.transcribe(&req(&audio, Speaker::Local))
                .expect("ok")
                .utterances[0]
                .text,
            "first line"
        );
        assert_eq!(
            m.transcribe(&req(&audio, Speaker::Remote))
                .expect("ok")
                .utterances[0]
                .text,
            "second line"
        );
        let third = m.transcribe(&req(&audio, Speaker::Local)).expect("ok");
        assert_ne!(third.utterances[0].text, "");
    }

    #[test]
    fn a_script_is_not_consumed_by_silence() {
        let m = MockTranscriber::scripted(["only line"]);
        let quiet = vec![0.0_f32; 16_000];
        m.transcribe(&req(&quiet, Speaker::Local)).expect("ok");
        let out = m
            .transcribe(&req(&tone(500, 0.5), Speaker::Local))
            .expect("ok");
        assert_eq!(out.utterances[0].text, "only line");
    }

    #[test]
    fn failure_is_an_error_not_a_panic() {
        let m = MockTranscriber::failing("model unloadable");
        match m.transcribe(&req(&tone(500, 0.5), Speaker::Local)) {
            Err(AiError::Transcription(r)) => assert_eq!(r, "model unloadable"),
            other => panic!("expected Transcription error, got {other:?}"),
        }
    }

    #[test]
    fn a_zero_sample_rate_is_rejected() {
        let m = MockTranscriber::new();
        let audio = tone(500, 0.5);
        let bad = TranscribeRequest {
            samples: &audio,
            sample_rate: 0,
            speaker: Speaker::Local,
            language: None,
        };
        assert!(matches!(m.transcribe(&bad), Err(AiError::InvalidAudio(_))));
    }

    #[test]
    fn a_forced_language_is_echoed_back() {
        let m = MockTranscriber::new().with_language("en");
        let audio = tone(500, 0.5);
        let out = m
            .transcribe(&TranscribeRequest {
                samples: &audio,
                sample_rate: 16_000,
                speaker: Speaker::Local,
                language: Some("de".into()),
            })
            .expect("ok");
        assert_eq!(out.language, "de");
    }

    #[test]
    fn the_model_id_is_reported_for_the_audit_trail() {
        let m = MockTranscriber::new().with_model_id("whisper-base-q5_1");
        assert_eq!(m.model_id(), "whisper-base-q5_1");
    }
}

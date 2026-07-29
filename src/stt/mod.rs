//! Speech-to-text engine interface.
//!
//! The crate owns segmentation and speaker attribution; an engine only has to
//! turn one mono 16 kHz buffer into timed text. That keeps the trait small
//! enough that swapping Parakeet for Whisper — or for a mock — changes
//! nothing above it.
//!
//! Real backends live behind Cargo features so the crate builds, tests and
//! ships without multi-gigabyte native dependencies:
//!
//! | Feature | Backend | Status per the model research |
//! |---|---|---|
//! | `stt-parakeet` | Parakeet-TDT-0.6B-v2 via `sherpa-rs` | recommended primary |
//! | `stt-whisper` | `whisper.cpp` via `whisper-rs` | secondary / back pocket |
//! | *(none)* | [`mock::MockTranscriber`] | deterministic, used by the tests |

pub mod mock;

#[cfg(feature = "stt-parakeet")]
pub mod parakeet;

#[cfg(feature = "stt-whisper")]
pub mod whisper;

pub use mock::MockTranscriber;

use crate::error::Result;
use crate::types::Speaker;

/// One piece of recognised speech.
///
/// **Timings are relative to the start of the buffer that was passed in**, not
/// to the call. The session applies the segment offset.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    pub start_ms: u32,
    pub end_ms: u32,
    pub text: String,
    /// `[0, 1]`. Engines that report nothing useful return `1.0`; this is not
    /// a calibrated probability and must not be shown as one.
    pub confidence: f32,
}

/// One unit of work for an engine.
#[derive(Debug)]
pub struct TranscribeRequest<'a> {
    /// Mono `f32` in `[-1.0, 1.0]`.
    pub samples: &'a [f32],
    /// Always [`crate::audio::TARGET_RATE_HZ`] in practice.
    pub sample_rate: u32,
    /// Which leg this came from. Engines generally ignore it; it is here so
    /// logs and mocks can be leg-aware.
    pub speaker: Speaker,
    /// Forced recognition language, or `None` to auto-detect.
    pub language: Option<String>,
}

/// What an engine gives back.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscribeOutput {
    pub utterances: Vec<Utterance>,
    /// Detected or echoed-back language tag.
    pub language: String,
}

impl TranscribeOutput {
    /// An empty result in `language` — the correct answer for silence.
    #[must_use]
    pub fn empty(language: impl Into<String>) -> Self {
        Self {
            utterances: Vec::new(),
            language: language.into(),
        }
    }
}

/// A speech-to-text engine.
///
/// Implementations must be `Send + Sync` — the session may drive them from a
/// worker thread — and must never panic, because everything above them is
/// reachable over FFI.
pub trait Transcriber: Send + Sync {
    /// Catalogue id of the loaded model, for the audit trail on the insight.
    fn model_id(&self) -> String;

    /// Recognise one buffer.
    ///
    /// # Errors
    /// [`crate::AiError::Transcription`] if the engine fails, or
    /// [`crate::AiError::InvalidAudio`] for a buffer it cannot accept.
    fn transcribe(&self, req: &TranscribeRequest<'_>) -> Result<TranscribeOutput>;
}

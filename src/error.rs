//! The single error type crossing every public boundary of this crate.
//!
//! Two hard rules, both driven by the fact that this crate is called over FFI
//! where a Rust unwind is undefined behaviour:
//!
//! 1. Every fallible public function returns [`AiError`]; none of them panic.
//! 2. No `unwrap`/`expect` on any input-derived path. Lock poisoning is
//!    recovered from rather than propagated (see [`crate::util::lock`]).

use std::fmt;

/// Everything that can go wrong inside `aria-ai-core`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Error))]
#[cfg_attr(feature = "uniffi", uniffi(flat_error))]
pub enum AiError {
    /// Configuration supplied by the host does not describe a usable engine.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A sample rate the audio front end cannot convert to 16 kHz.
    #[error("unsupported sample rate: {rate} Hz")]
    UnsupportedSampleRate { rate: u32 },

    /// Audio arrived that violates the input contract (empty, mismatched, …).
    #[error("invalid audio: {0}")]
    InvalidAudio(String),

    /// The session is not in a state where the requested call is legal.
    #[error("session is {actual}, expected {expected}")]
    InvalidState { expected: String, actual: String },

    /// Capture ran past `max_capture_secs` and was sealed early.
    #[error("capture limit of {limit_secs}s reached")]
    CaptureLimitReached { limit_secs: u32 },

    /// No catalog entry with this id.
    #[error("unknown model id: {0}")]
    UnknownModel(String),

    /// The model is in the catalog but not present on disk.
    #[error("model not installed: {0}")]
    ModelNotInstalled(String),

    /// The model needs more device than this device has.
    #[error("model {id} needs device tier {required}, this device is {actual}")]
    DeviceTooSmall {
        id: String,
        required: String,
        actual: String,
    },

    /// Free-space preflight failed.
    #[error("insufficient disk space: need {needed_bytes} bytes, {available_bytes} available")]
    InsufficientSpace {
        needed_bytes: u64,
        available_bytes: u64,
    },

    /// The download host is not on the compile-time allowlist.
    #[error("host not permitted: {0}")]
    HostNotAllowed(String),

    /// Transport-level download failure.
    #[error("download failed: {0}")]
    Download(String),

    /// A download was cancelled cooperatively. The `.part` file is kept.
    #[error("download cancelled")]
    DownloadCancelled,

    /// SHA-256 of the downloaded bytes did not match the bundled catalog.
    #[error("integrity check failed for {id}: expected {expected}, got {actual}")]
    IntegrityMismatch {
        id: String,
        expected: String,
        actual: String,
    },

    /// Filesystem failure, with the offending path.
    #[error("i/o error at {path}: {message}")]
    Io { path: String, message: String },

    /// The STT engine failed.
    #[error("transcription failed: {0}")]
    Transcription(String),

    /// The LLM engine failed.
    #[error("summarization failed: {0}")]
    Summarization(String),

    /// The LLM produced something that is not the agreed JSON schema.
    #[error("model output did not match the summary schema: {0}")]
    InvalidModelOutput(String),

    /// Too little speech to summarize honestly (see design spec §4.4 "floor").
    #[error("transcript has only {speech_secs}s of speech; minimum is {minimum_secs}s")]
    TranscriptTooShort {
        speech_secs: u32,
        minimum_secs: u32,
    },

    /// The requested capability is compiled out.
    #[error("feature `{0}` is not enabled in this build")]
    FeatureNotEnabled(String),

    /// An invariant inside the crate was violated. Never expected; never panics.
    #[error("internal error: {0}")]
    Internal(String),
}

impl AiError {
    /// Build an [`AiError::Io`] from a path and a `std::io::Error`.
    pub fn io(path: impl AsRef<std::path::Path>, err: &std::io::Error) -> Self {
        Self::Io {
            path: path.as_ref().display().to_string(),
            message: err.to_string(),
        }
    }

    /// Build an [`AiError::Io`] from a path and a message.
    pub fn io_msg(path: impl AsRef<std::path::Path>, message: impl fmt::Display) -> Self {
        Self::Io {
            path: path.as_ref().display().to_string(),
            message: message.to_string(),
        }
    }

    /// True when retrying the same operation could plausibly succeed.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Download(_) | Self::Io { .. })
    }
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, AiError>;

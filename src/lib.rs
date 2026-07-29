//! # aria-ai-core
//!
//! On-device call transcription and summarization for the Aria softphone
//! family. Everything runs locally; the only network capability in the whole
//! crate is the model downloader, and it is restricted to a compile-time host
//! allowlist ([`models::ALLOWED_HOSTS`]).
//!
//! ## Who consumes this
//!
//! * **Desktop** (`softphone`, Tauri v2) — a plain Rust dependency. Tauri is
//!   Rust; there is no FFI in the way.
//! * **iOS / Android** — through `aria-mobile-core`'s `UniFFI` surface. The
//!   optional `uniffi` feature exposes the types and a facade object directly;
//!   without it, not a byte of `UniFFI` is compiled.
//!
//! That is why the API is plain Rust with `UniFFI` layered on top as
//! `#[cfg_attr]` attributes and one gated module, rather than a `UDL`-first
//! design: the desktop must not pay for the mobile binding layer.
//!
//! ## Layout
//!
//! | Module | What it owns |
//! |---|---|
//! | [`audio`] | ring buffers, band-limited resampling to 16 kHz, VAD, speaker-attributed segmentation |
//! | [`models`] | catalogue, device gating, resumable/verified download, on-disk store |
//! | [`stt`] | the [`stt::Transcriber`] trait, a deterministic mock, feature-gated real backends |
//! | [`llm`] | the [`llm::Summarizer`] trait, prompts + GBNF grammar, chunking, the validating map-reduce pipeline |
//! | [`session`] | [`session::CallSession`] — start/stop, feed frames, transcript, summary |
//! | [`engine`] | [`engine::AiEngine`] — the one object each platform binds to |
//!
//! ## Building without the heavy dependencies
//!
//! Default features pull in nothing larger than `reqwest`. The inference
//! backends are opt-in:
//!
//! ```text
//! cargo test                              # no models, no network, no C++
//! cargo build --features stt-parakeet      # sherpa-onnx / Parakeet-TDT
//! cargo build --features stt-whisper       # whisper.cpp
//! cargo build --features llm-llama         # llama.cpp
//! cargo build --features uniffi            # Swift/Kotlin scaffolding
//! ```
//!
//! ## Nothing panics
//!
//! This crate is called across an FFI boundary where a Rust unwind is
//! undefined behaviour. Every fallible public function returns [`AiError`],
//! there is no `unwrap`/`expect` on any input-derived path, and mutex
//! poisoning is recovered from rather than propagated.
//!
//! ## Example
//!
//! ```
//! use aria_ai_core::{
//!     llm::{mock::MockSummarizer, prompt::CallContext, pipeline::SummaryConfig},
//!     session::{CallSession, SessionConfig},
//!     stt::MockTranscriber,
//! };
//!
//! # fn main() -> Result<(), aria_ai_core::AiError> {
//! let session = CallSession::start(SessionConfig::new("call-42"))?;
//!
//! // 20 ms of 8 kHz PCMU from each leg, exactly as `PlatformAudioBridge`
//! // delivers it.
//! session.push_remote(&[0i16; 160], 8_000)?;
//! session.push_local(&[0i16; 160], 8_000)?;
//! session.pump()?;
//!
//! session.stop()?;
//! let transcript = session.transcribe(&MockTranscriber::new())?;
//! assert_eq!(transcript.call_id, "call-42");
//! # Ok(())
//! # }
//! ```

#![warn(clippy::pedantic, clippy::nursery)]
// Every module, type, trait, function and constant is documented. Field-level
// docs are present where a field is not self-describing; `missing_docs` is not
// enabled because it would demand a comment on `pub start_ms: u32`.
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]
// Panicking is undefined behaviour across FFI. Tests may still assert.
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]
// Audio DSP and token arithmetic convert between integer and float widths on
// almost every line. Every such site in this crate is bounded by construction
// (sample counts, millisecond offsets, token counts, filter tap indices) or
// uses a saturating/`try_from` conversion where it is not. Denying these four
// would mean an `#[allow]` on roughly a third of the functions here, which
// hides the sites that genuinely deserve attention.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
// Guards in this crate protect plain data and are held for a memcpy or a
// clone. Tightening their scope further obscures more than it buys.
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::module_name_repetitions)]

pub mod audio;
pub mod engine;
#[cfg(feature = "insights")]
pub mod insights;
pub mod error;
pub mod llm;
pub mod models;
pub mod session;
pub mod stt;
pub mod types;

mod util;

#[cfg(feature = "uniffi")]
pub mod ffi;

pub use error::{AiError, Result};

pub use audio::{
    segment_call, AudioSegment, CallAudioTap, CapturedAudio, SegmentationConfig, SincResampler,
    TapConfig, VadConfig, TARGET_RATE_HZ,
};
pub use engine::{AiConfig, AiEngine};
pub use llm::{
    pipeline::SummaryConfig, prompt::CallContext, ChunkConfig, GenerationRequest, Summarizer,
};
pub use models::{
    Catalog, DeviceCapability, DeviceTier, DownloadProgress, DownloadState, ModelDescriptor,
    ModelKind, ModelStore, Platform,
};
pub use session::{CallSession, DrainHandle, SessionConfig};
pub use stt::{TranscribeOutput, TranscribeRequest, Transcriber, Utterance};
pub use types::{
    ActionItem, ActionOwner, CallInsight, InsightStatus, Speaker, Summary, Transcript,
    TranscriptSegment,
};

/// The crate version, for the audit trail on a stored insight.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

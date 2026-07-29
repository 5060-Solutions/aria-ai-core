//! Summarization: the engine interface, the prompt/schema layer, the
//! chunking strategy for long calls, and the validating map-reduce pipeline.
//!
//! The trait deliberately stops at "give me text for this prompt". Everything
//! that makes the feature correct — windowing, retry, JSON validation,
//! clamping, the degraded fallback — is shared Rust above it, so a mock and
//! `llama.cpp` behave identically from the caller's point of view.

pub mod chunking;
pub mod mock;
pub mod pipeline;
pub mod prompt;

#[cfg(feature = "llm-llama")]
pub mod llama;

pub use chunking::{plan_windows, ChunkConfig, Window};
pub use mock::MockSummarizer;
pub use pipeline::{summarize, SummaryConfig, SummaryOutcome};
pub use prompt::{CallContext, SUMMARY_GRAMMAR, SYSTEM_PROMPT};

use crate::error::Result;

/// Sampling defaults from design spec §4.2 — low temperature, because this is
/// extraction, not creative writing.
pub const DEFAULT_TEMPERATURE: f32 = 0.2;
/// Nucleus sampling cutoff.
pub const DEFAULT_TOP_P: f32 = 0.9;
/// Mild repetition penalty.
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.05;
/// Cap on generated tokens for the reduce/single-pass step.
pub const DEFAULT_MAX_TOKENS: u32 = 512;

/// One generation request.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationRequest {
    pub system: String,
    pub user: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    /// GBNF grammar constraining the output. Engines that cannot constrain
    /// decoding must ignore this; the Rust-side validator catches the rest.
    pub grammar: Option<String>,
}

impl GenerationRequest {
    /// A request with the spec's sampling defaults.
    #[must_use]
    pub fn new(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self {
            system: system.into(),
            user: user.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            repeat_penalty: DEFAULT_REPEAT_PENALTY,
            grammar: None,
        }
    }

    /// Constrain decoding with a GBNF grammar.
    #[must_use]
    pub fn with_grammar(mut self, grammar: impl Into<String>) -> Self {
        self.grammar = Some(grammar.into());
        self
    }

    /// Cap generated tokens.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

/// A text-generation engine used for summarization.
///
/// Implementations must be `Send + Sync` and must never panic.
pub trait Summarizer: Send + Sync {
    /// Catalogue id of the loaded model, for the audit trail on the insight.
    fn model_id(&self) -> String;

    /// Usable context window in tokens. Drives [`ChunkConfig`].
    fn context_tokens(&self) -> usize;

    /// Generate a completion.
    ///
    /// # Errors
    /// [`crate::AiError::Summarization`] if the engine fails.
    fn complete(&self, req: &GenerationRequest) -> Result<String>;
}

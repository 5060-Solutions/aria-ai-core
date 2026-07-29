//! `llama.cpp` backend, behind the `llm-llama` feature.
//!
//! This is the runtime the design spec picks specifically for **GBNF
//! grammar-constrained decoding** (§4.2): with a 1–4B model, malformed JSON is
//! the single largest failure mode, and constraining the sampler removes it
//! outright rather than retrying around it. [`GenerationRequest::grammar`] is
//! honoured here; the Rust-side validator in [`crate::llm::pipeline`] still
//! runs, because the grammar fixes the shape and not the content.
//!
//! Structure notes:
//!
//! * The backend is a process-wide singleton — `llama_backend_init` may only
//!   run once — so it lives in a `OnceLock` and is never dropped.
//! * A fresh `LlamaContext` is built per generation. `LlamaContext<'a>` borrows
//!   the model, so caching one would make this struct self-referential; a
//!   summarization does a handful of generations per call, and a KV-cache
//!   allocation is cheap next to the decode itself.
//! * Nothing here panics: every `llama-cpp-2` error becomes an [`AiError`].

use std::num::NonZeroU32;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::error::{AiError, Result};
use crate::llm::{GenerationRequest, Summarizer};
use crate::models::capability::DeviceCapability;
use crate::models::catalog::ModelDescriptor;

/// Context window used when the catalogue entry does not state one.
const DEFAULT_CONTEXT_TOKENS: u32 = 4_096;

/// Root rule of [`crate::llm::prompt::SUMMARY_GRAMMAR`].
const GRAMMAR_ROOT: &str = "root";

/// Tokens per decode batch. Prompt ingestion is chunked to this.
const BATCH_TOKENS: u32 = 512;

/// `penalty_last_n` for the repetition penalty sampler.
const PENALTY_WINDOW: i32 = 64;

/// `llama_backend_init` is process-wide and must not run twice. The value is
/// intentionally never dropped: a static is not destructed, which is exactly
/// the lifetime llama.cpp wants.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Initialise (once) and borrow the process-wide llama.cpp backend.
fn backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    // A lost race here means someone else initialised it; take theirs.
    match LlamaBackend::init() {
        Ok(b) => Ok(BACKEND.get_or_init(|| b)),
        Err(e) => BACKEND
            .get()
            .ok_or_else(|| AiError::Summarization(format!("llama backend init: {e}"))),
    }
}

/// A loaded GGUF model.
pub struct LlamaSummarizer {
    model_id: String,
    context_tokens: u32,
    threads: i32,
    /// `LlamaModel` is `Send + Sync`; contexts are created per generation.
    model: LlamaModel,
}

impl std::fmt::Debug for LlamaSummarizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaSummarizer")
            .field("model_id", &self.model_id)
            .field("context_tokens", &self.context_tokens)
            .field("threads", &self.threads)
            .finish_non_exhaustive()
    }
}

impl LlamaSummarizer {
    /// Load a GGUF model from disk.
    ///
    /// # Errors
    /// [`AiError::Summarization`] if the backend cannot start or the file
    /// cannot be loaded as a GGUF model.
    pub fn load(model: &ModelDescriptor, path: &str, cap: &DeviceCapability) -> Result<Self> {
        // `LlamaModel::load_from_file` opens with a `debug_assert!` that the
        // path exists, so in a debug build a missing file *panics inside the
        // dependency*. A panic is undefined behaviour across FFI, so the
        // existence check happens here, before llama.cpp is ever reached.
        if !std::path::Path::new(path).is_file() {
            return Err(AiError::ModelNotInstalled(format!(
                "{}: no model file at {path}",
                model.id
            )));
        }

        let backend = backend()?;

        let mut params = LlamaModelParams::default();
        if cap.has_gpu_backend {
            // Offload everything the device will take; llama.cpp clamps this
            // down to the layer count it actually has.
            params = params.with_n_gpu_layers(u32::MAX);
        } else {
            params = params.with_n_gpu_layers(0);
        }

        let loaded = LlamaModel::load_from_file(backend, path, &params)
            .map_err(|e| AiError::Summarization(format!("llama: loading {path}: {e}")))?;

        // The catalogue states the window we intend to use, which is usually
        // smaller than what the model was trained for. Never exceed training.
        let trained = loaded.n_ctx_train();
        let wanted = model.context_tokens.unwrap_or(DEFAULT_CONTEXT_TOKENS);
        let context_tokens = wanted.min(trained).max(1);

        Ok(Self {
            model_id: model.id.clone(),
            context_tokens,
            threads: i32::try_from(cap.inference_threads()).unwrap_or(1).max(1),
            model: loaded,
        })
    }

    /// Render the system/user pair through the model's own chat template,
    /// falling back to a plain concatenation for a model that ships none.
    fn render_prompt(&self, req: &GenerationRequest) -> String {
        let messages = [
            LlamaChatMessage::new("system".to_string(), req.system.clone()),
            LlamaChatMessage::new("user".to_string(), req.user.clone()),
        ];
        let fallback = || format!("{}\n\n{}\n\n", req.system, req.user);

        // A null byte in a prompt, or a model with no baked-in template, is a
        // reason to fall back — never a reason to fail the summary.
        let (Ok(system), Ok(user)) = (&messages[0], &messages[1]) else {
            return fallback();
        };
        let Ok(template) = self.model.chat_template(None) else {
            return fallback();
        };
        self.model
            .apply_chat_template(&template, &[system.clone(), user.clone()], true)
            .unwrap_or_else(|_| fallback())
    }

    /// Build the sampler chain: grammar first so it constrains everything
    /// downstream, then the spec's sampling defaults.
    fn sampler(&self, req: &GenerationRequest) -> LlamaSampler {
        let mut chain = Vec::new();
        if let Some(grammar) = req.grammar.as_deref() {
            match LlamaSampler::grammar(&self.model, grammar, GRAMMAR_ROOT) {
                Ok(g) => chain.push(g),
                // An unparseable grammar degrades to free decoding; the
                // pipeline's JSON validator is the backstop.
                Err(e) => log::warn!("llama: ignoring bad grammar: {e}"),
            }
        }
        chain.push(LlamaSampler::penalties(
            PENALTY_WINDOW,
            req.repeat_penalty,
            0.0,
            0.0,
        ));
        chain.push(LlamaSampler::top_p(req.top_p, 1));
        chain.push(LlamaSampler::temp(req.temperature));
        chain.push(LlamaSampler::dist(0));
        LlamaSampler::chain_simple(chain)
    }
}

impl Summarizer for LlamaSummarizer {
    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn context_tokens(&self) -> usize {
        self.context_tokens as usize
    }

    fn complete(&self, req: &GenerationRequest) -> Result<String> {
        let backend = backend()?;
        let prompt = self.render_prompt(req);

        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| AiError::Summarization(format!("llama: tokenizing: {e}")))?;

        // Reserve room for the answer. The chunker above already sizes the
        // prompt, but a host can call `complete` directly.
        let max_new = req.max_tokens.max(1);
        let needed = u32::try_from(tokens.len())
            .unwrap_or(u32::MAX)
            .saturating_add(max_new);
        if needed > self.context_tokens {
            return Err(AiError::Summarization(format!(
                "prompt plus {max_new} new tokens needs {needed} of a {} token context",
                self.context_tokens
            )));
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.context_tokens))
            .with_n_batch(BATCH_TOKENS)
            .with_n_threads(self.threads)
            .with_n_threads_batch(self.threads);
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| AiError::Summarization(format!("llama: context: {e}")))?;

        // ── ingest the prompt ────────────────────────────────────────────────
        let batch_cap = BATCH_TOKENS as usize;
        let mut batch = LlamaBatch::new(batch_cap, 1);
        let last = tokens.len() - 1;
        for (chunk_start, chunk) in tokens.chunks(batch_cap).enumerate().map(|(i, c)| (i * batch_cap, c)) {
            batch.clear();
            for (offset, token) in chunk.iter().enumerate() {
                let pos = chunk_start + offset;
                let pos_i32 = i32::try_from(pos)
                    .map_err(|_| AiError::Summarization("prompt too long".to_string()))?;
                batch
                    .add(*token, pos_i32, &[0], pos == last)
                    .map_err(|e| AiError::Summarization(format!("llama: batch: {e}")))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| AiError::Summarization(format!("llama: decode: {e}")))?;
        }

        // ── decode the answer ────────────────────────────────────────────────
        let mut sampler = self.sampler(req);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let mut pos = i32::try_from(tokens.len())
            .map_err(|_| AiError::Summarization("prompt too long".to_string()))?;

        for _ in 0..max_new {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if self.model.is_eog_token(token) {
                break;
            }
            sampler.accept(token);

            match self
                .model
                .token_to_piece(token, &mut decoder, false, None)
            {
                Ok(piece) => out.push_str(&piece),
                // A token that is not representable on its own is dropped
                // rather than aborting a summary that is otherwise fine.
                Err(e) => log::debug!("llama: undecodable token: {e}"),
            }

            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .map_err(|e| AiError::Summarization(format!("llama: batch: {e}")))?;
            ctx.decode(&mut batch)
                .map_err(|e| AiError::Summarization(format!("llama: decode: {e}")))?;
            pos = pos.saturating_add(1);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desktop() -> DeviceCapability {
        DeviceCapability {
            platform: crate::models::Platform::MacOs,
            total_ram_mb: 16_384,
            available_ram_mb: 8_192,
            cpu_cores: 8,
            is_64bit: true,
            has_gpu_backend: false,
        }
    }

    fn llm_descriptor() -> ModelDescriptor {
        crate::models::catalog::Catalog::bundled()
            .expect("catalog")
            .models
            .iter()
            .find(|m| m.runtime == crate::models::ModelRuntime::LlamaCpp)
            .expect("an llm model in the catalogue")
            .clone()
    }

    #[test]
    fn the_backend_initialises_once_and_is_reusable() {
        let a = backend().expect("backend");
        let b = backend().expect("backend again");
        assert!(std::ptr::eq(a, b), "the backend must be a singleton");
    }

    #[test]
    fn a_missing_model_file_is_an_error_not_a_panic() {
        // Regression guard: `llama-cpp-2`'s `load_from_file` `debug_assert!`s
        // that the path exists, so without our own check this panics.
        let err = LlamaSummarizer::load(&llm_descriptor(), "/nonexistent/model.gguf", &desktop())
            .expect_err("loading a missing file must fail");
        assert!(matches!(err, AiError::ModelNotInstalled(_)), "{err:?}");
    }

    #[test]
    fn a_file_that_is_not_a_gguf_model_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not-a-model.gguf");
        std::fs::write(&path, b"definitely not GGUF").expect("write");

        let err = LlamaSummarizer::load(
            &llm_descriptor(),
            path.to_str().expect("utf8"),
            &desktop(),
        )
        .expect_err("garbage must not load");
        assert!(matches!(err, AiError::Summarization(_)), "{err:?}");
    }
}

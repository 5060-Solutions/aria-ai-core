//! `whisper.cpp` backend, behind the `stt-whisper` feature.
//!
//! Per the model research this is the secondary ("back pocket") engine: it is
//! the one that ships as a single GGML file, runs everywhere, and has a known
//! narrowband word-error rate. [`super::parakeet`] is the recommended primary.
//!
//! Three things this module is responsible for, none of which belong in the
//! shared code above it:
//!
//! * **Serialising access.** A `whisper_state` is not re-entrant, so one lives
//!   behind a `Mutex` and a call waits rather than corrupting it.
//! * **The 1-second floor.** `whisper.cpp` needs at least one second of audio
//!   or it returns nothing useful; short VAD segments are zero-padded.
//! * **Never panicking.** Every `whisper-rs` error becomes an [`AiError`], and
//!   a segment whose text is not valid UTF-8 is skipped rather than unwrapped.

use std::sync::{Mutex, Once};

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::audio::TARGET_RATE_HZ;
use crate::error::{AiError, Result};
use crate::models::capability::DeviceCapability;
use crate::models::catalog::ModelDescriptor;
use crate::stt::{TranscribeOutput, TranscribeRequest, Transcriber, Utterance};
use crate::util::lock;

/// `whisper.cpp` reports segment times in hundredths of a second.
const CENTISECOND_MS: i64 = 10;

/// Shortest buffer `whisper.cpp` handles sensibly. Anything less is padded.
const MIN_SAMPLES: usize = TARGET_RATE_HZ as usize;

/// Language tag used when detection produced nothing.
const UNKNOWN_LANGUAGE: &str = "und";

/// Whisper's own `no_speech` probability is the only per-segment confidence
/// signal it exposes without token timestamps.
const DEFAULT_CONFIDENCE: f32 = 1.0;

static LOGGING: Once = Once::new();

/// A loaded `whisper.cpp` model.
pub struct WhisperTranscriber {
    model_id: String,
    threads: std::ffi::c_int,
    /// `whisper_state` is `Send + Sync` but not re-entrant: serialise calls.
    /// The state owns an `Arc` to the context, so nothing borrows here.
    state: Mutex<WhisperState>,
}

impl std::fmt::Debug for WhisperTranscriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperTranscriber")
            .field("model_id", &self.model_id)
            .field("threads", &self.threads)
            .finish_non_exhaustive()
    }
}

impl WhisperTranscriber {
    /// Load a GGML whisper model from disk.
    ///
    /// # Errors
    /// [`AiError::Transcription`] if `whisper.cpp` cannot open the file or
    /// allocate a state for it.
    pub fn load(model: &ModelDescriptor, path: &str, cap: &DeviceCapability) -> Result<Self> {
        // whisper.cpp writes to stderr by default; route it into `log`.
        LOGGING.call_once(whisper_rs::install_logging_hooks);

        let mut params = WhisperContextParameters::default();
        // The host decides whether a GPU backend is usable on this device;
        // the compiled-in default only knows what was linked.
        params.use_gpu(cap.has_gpu_backend);

        let ctx = WhisperContext::new_with_params(path, params)
            .map_err(|e| AiError::Transcription(format!("whisper: loading {path}: {e}")))?;
        let state = ctx
            .create_state()
            .map_err(|e| AiError::Transcription(format!("whisper: creating state: {e}")))?;

        Ok(Self {
            model_id: model.id.clone(),
            threads: std::ffi::c_int::try_from(cap.inference_threads()).unwrap_or(1).max(1),
            state: Mutex::new(state),
        })
    }
}

impl Transcriber for WhisperTranscriber {
    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn transcribe(&self, req: &TranscribeRequest<'_>) -> Result<TranscribeOutput> {
        if req.sample_rate != TARGET_RATE_HZ {
            return Err(AiError::InvalidAudio(format!(
                "whisper needs {TARGET_RATE_HZ} Hz, got {}",
                req.sample_rate
            )));
        }
        let fallback_language = req
            .language
            .clone()
            .unwrap_or_else(|| UNKNOWN_LANGUAGE.to_string());
        if req.samples.is_empty() {
            return Ok(TranscribeOutput::empty(fallback_language));
        }

        // whisper.cpp silently returns nothing for buffers under a second.
        let padded: Vec<f32>;
        let samples = if req.samples.len() < MIN_SAMPLES {
            padded = req
                .samples
                .iter()
                .copied()
                .chain(std::iter::repeat_n(0.0, MIN_SAMPLES - req.samples.len()))
                .collect();
            &padded[..]
        } else {
            req.samples
        };

        // Greedy rather than beam search: this runs once per VAD segment on a
        // phone, and beam search costs several times the CPU for a small win.
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads);
        params.set_translate(false);
        params.set_no_context(true);
        params.set_suppress_blank(true);
        params.set_single_segment(false);
        // Nothing may reach stdout: this is a library inside a phone app.
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        match forced_language(req.language.as_deref())? {
            Some(lang) => params.set_language(Some(lang)),
            None => params.set_detect_language(true),
        }

        let mut state = lock(&self.state);
        state
            .full(params, samples)
            .map_err(|e| AiError::Transcription(format!("whisper: {e}")))?;

        let language = whisper_rs::get_lang_str(state.full_lang_id_from_state())
            .map_or_else(|| fallback_language.clone(), ToString::to_string);

        let mut utterances = Vec::new();
        for segment in state.as_iter() {
            // Invalid UTF-8 out of the model loses one segment, not the call.
            let Ok(text) = segment.to_str_lossy() else {
                continue;
            };
            let text = text.trim().to_string();
            if text.is_empty() {
                continue;
            }
            utterances.push(Utterance {
                start_ms: to_ms(segment.start_timestamp()),
                end_ms: to_ms(segment.end_timestamp()),
                text,
                // `no_speech_probability` is the only confidence signal whisper
                // gives without token timestamps. It is not calibrated.
                confidence: (DEFAULT_CONFIDENCE - segment.no_speech_probability())
                    .clamp(0.0, 1.0),
            });
        }

        Ok(TranscribeOutput {
            utterances,
            language,
        })
    }
}

/// Validate a host-supplied language tag before it reaches `whisper.cpp`.
///
/// `whisper-rs::set_language` builds a `CString` with `.expect()`, so an
/// interior NUL would panic *inside the dependency*, and a panic across FFI is
/// undefined behaviour. An empty or all-whitespace tag means "auto-detect".
fn forced_language(language: Option<&str>) -> Result<Option<&str>> {
    match language {
        None => Ok(None),
        Some(lang) if lang.trim().is_empty() => Ok(None),
        Some(lang) if lang.contains('\0') => Err(AiError::Config(
            "language tag contains a NUL byte".to_string(),
        )),
        Some(lang) => Ok(Some(lang)),
    }
}

/// Centiseconds from `whisper.cpp` to milliseconds, saturating.
fn to_ms(centiseconds: i64) -> u32 {
    u32::try_from(centiseconds.max(0).saturating_mul(CENTISECOND_MS)).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centisecond_conversion_saturates_instead_of_panicking() {
        assert_eq!(to_ms(0), 0);
        assert_eq!(to_ms(123), 1_230);
        assert_eq!(to_ms(-5), 0);
        assert_eq!(to_ms(i64::MAX), u32::MAX);
    }

    #[test]
    fn a_language_tag_with_a_nul_byte_is_rejected_before_whisper_sees_it() {
        // Regression guard: `whisper-rs::set_language` panics on an interior
        // NUL, and that panic would cross an FFI boundary.
        let err = forced_language(Some("en\0evil")).expect_err("must be rejected");
        assert!(matches!(err, AiError::Config(_)), "{err:?}");
    }

    #[test]
    fn an_absent_or_blank_language_means_auto_detect() {
        assert_eq!(forced_language(None).expect("ok"), None);
        assert_eq!(forced_language(Some("   ")).expect("ok"), None);
        assert_eq!(forced_language(Some("en")).expect("ok"), Some("en"));
    }

    #[test]
    fn a_missing_model_file_is_an_error_not_a_panic() {
        let model = crate::models::catalog::Catalog::bundled()
            .expect("catalog")
            .models
            .iter()
            .find(|m| m.runtime == crate::models::ModelRuntime::WhisperCpp)
            .expect("a whisper model in the catalogue")
            .clone();
        let cap = DeviceCapability {
            platform: crate::models::Platform::MacOs,
            total_ram_mb: 16_384,
            available_ram_mb: 8_192,
            cpu_cores: 8,
            is_64bit: true,
            has_gpu_backend: false,
        };
        let err = WhisperTranscriber::load(&model, "/nonexistent/whisper.bin", &cap)
            .expect_err("loading a missing file must fail");
        assert!(matches!(err, AiError::Transcription(_)), "{err:?}");
    }
}

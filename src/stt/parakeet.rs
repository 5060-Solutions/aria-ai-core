//! Parakeet-TDT via `sherpa-onnx`, behind the `stt-parakeet` feature.
//!
//! The model research picks this as the primary STT engine: materially better
//! narrowband accuracy than whisper at a comparable footprint.
//!
//! ## Packaging: this needs a directory, not a file
//!
//! Unlike `whisper.cpp` (one GGML file) a sherpa-onnx transducer is four
//! files — `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt` — and
//! upstream distributes them as a `.tar.bz2`. [`crate::models::ModelStore`]
//! installs exactly the bytes it downloaded and **does not unpack archives**,
//! so until it learns to, this backend must be pointed at a directory that the
//! host has already extracted. [`ParakeetTranscriber::load`] says so in the
//! error rather than failing obscurely inside `sherpa-onnx`.
//!
//! ## Timings
//!
//! `sherpa-onnx`'s offline API returns text with no word timings, so a
//! transcription here is one [`Utterance`] spanning the buffer it was given.
//! That is sufficient because this crate does its own VAD segmentation: the
//! session applies the segment's own start/end (see [`crate::session`]).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};

use crate::audio::TARGET_RATE_HZ;
use crate::error::{AiError, Result};
use crate::models::capability::DeviceCapability;
use crate::models::catalog::ModelDescriptor;
use crate::stt::{TranscribeOutput, TranscribeRequest, Transcriber, Utterance};
use crate::util::{lock, samples_to_ms};

/// The four files a sherpa-onnx transducer needs, relative to the model dir.
// These must match the file names the catalogue actually downloads. The
// catalogue ships the int8 variants — that is the whole point of the
// "-int8" model — while these constants named the float32 files, so the
// primary STT engine could never load a model installed through the normal
// path. Nothing caught it because loading is only reachable with
// stt-parakeet enabled, which CI cannot build.
const ENCODER: &str = "encoder.int8.onnx";
const DECODER: &str = "decoder.int8.onnx";
const JOINER: &str = "joiner.int8.onnx";
const TOKENS: &str = "tokens.txt";

/// Filterbank dimension Parakeet was trained with.
const FEATURE_DIM: i32 = 80;

/// `sherpa-onnx` reports no per-word confidence; the trait's documented
/// "nothing useful to report" value.
const DEFAULT_CONFIDENCE: f32 = 1.0;

/// Language tag used when nothing was forced. Parakeet-TDT-0.6B-v2 is English.
const DEFAULT_LANGUAGE: &str = "en";

/// A loaded Parakeet-TDT model.
pub struct ParakeetTranscriber {
    model_id: String,
    /// `transcribe` takes `&mut self`; the trait hands out `&self`.
    recognizer: Mutex<TransducerRecognizer>,
}

impl std::fmt::Debug for ParakeetTranscriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParakeetTranscriber")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl ParakeetTranscriber {
    /// Load a Parakeet transducer from an **extracted model directory**.
    ///
    /// # Errors
    /// [`AiError::Config`] if `path` is not a directory holding the four
    /// expected files (which is the case when the store still holds the raw
    /// `.tar.bz2`), or [`AiError::Transcription`] if `sherpa-onnx` refuses it.
    pub fn load(model: &ModelDescriptor, path: &str, cap: &DeviceCapability) -> Result<Self> {
        let dir = model_dir(Path::new(path))?;

        let config = TransducerConfig {
            encoder: file_in(&dir, ENCODER)?,
            decoder: file_in(&dir, DECODER)?,
            joiner: file_in(&dir, JOINER)?,
            tokens: file_in(&dir, TOKENS)?,
            num_threads: i32::try_from(cap.inference_threads()).unwrap_or(1).max(1),
            sample_rate: i32::try_from(TARGET_RATE_HZ).unwrap_or(16_000),
            feature_dim: FEATURE_DIM,
            decoding_method: "greedy_search".to_string(),
            model_type: "transducer".to_string(),
            ..TransducerConfig::default()
        };

        let recognizer = TransducerRecognizer::new(config).map_err(|e| {
            AiError::Transcription(format!("sherpa-onnx: loading {}: {e}", dir.display()))
        })?;

        Ok(Self {
            model_id: model.id.clone(),
            recognizer: Mutex::new(recognizer),
        })
    }
}

impl Transcriber for ParakeetTranscriber {
    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn transcribe(&self, req: &TranscribeRequest<'_>) -> Result<TranscribeOutput> {
        if req.sample_rate != TARGET_RATE_HZ {
            return Err(AiError::InvalidAudio(format!(
                "parakeet needs {TARGET_RATE_HZ} Hz, got {}",
                req.sample_rate
            )));
        }
        let language = req
            .language
            .clone()
            .unwrap_or_else(|| DEFAULT_LANGUAGE.to_string());
        if req.samples.is_empty() {
            return Ok(TranscribeOutput::empty(language));
        }

        let text = lock(&self.recognizer)
            .transcribe(req.sample_rate, req.samples)
            .trim()
            .to_string();
        if text.is_empty() {
            return Ok(TranscribeOutput::empty(language));
        }

        let end_ms = u32::try_from(samples_to_ms(req.samples.len(), req.sample_rate))
            .unwrap_or(u32::MAX);

        Ok(TranscribeOutput {
            utterances: vec![Utterance {
                start_ms: 0,
                end_ms,
                text,
                confidence: DEFAULT_CONFIDENCE,
            }],
            language,
        })
    }
}

/// Resolve the directory holding the four model files, with an error that
/// names the actual problem when the store holds an un-extracted archive.
fn model_dir(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    // Conventional layout: `<store>/parakeet.tar.bz2` extracted alongside it.
    let stripped = path.with_extension("");
    let unpacked = stripped.with_extension("");
    if unpacked.is_dir() {
        return Ok(unpacked);
    }
    Err(AiError::Config(format!(
        "sherpa-onnx needs an extracted model directory containing {ENCODER}, \
         {DECODER}, {JOINER} and {TOKENS}; {} is not one. The model store does \
         not unpack archives yet — extract it host-side for now.",
        path.display()
    )))
}

/// Path of a required file inside the model directory, as a `String` because
/// that is what `sherpa-rs` takes.
fn file_in(dir: &Path, name: &str) -> Result<String> {
    let p = dir.join(name);
    if !p.is_file() {
        return Err(AiError::Config(format!(
            "missing {name} in model directory {}",
            dir.display()
        )));
    }
    p.to_str().map(ToString::to_string).ok_or_else(|| {
        AiError::Config(format!("model path is not valid UTF-8: {}", p.display()))
    })
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

    fn parakeet_descriptor() -> ModelDescriptor {
        crate::models::catalog::Catalog::bundled()
            .expect("catalog")
            .models
            .iter()
            .find(|m| m.id.starts_with("parakeet"))
            .expect("a parakeet model in the catalogue")
            .clone()
    }

    #[test]
    fn an_unextracted_archive_is_a_clear_config_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("parakeet-tdt-0.6b-v2-int8.tar.bz2");
        std::fs::write(&archive, b"not really an archive").expect("write");

        let err = ParakeetTranscriber::load(
            &parakeet_descriptor(),
            archive.to_str().expect("utf8"),
            &desktop(),
        )
        .expect_err("an archive is not a model directory");
        match err {
            AiError::Config(m) => assert!(m.contains("extracted model directory"), "{m}"),
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn a_directory_missing_a_required_file_names_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(ENCODER), b"x").expect("write");

        let err = ParakeetTranscriber::load(
            &parakeet_descriptor(),
            dir.path().to_str().expect("utf8"),
            &desktop(),
        )
        .expect_err("an incomplete directory must fail");
        match err {
            AiError::Config(m) => assert!(m.contains(DECODER), "{m}"),
            other => panic!("wrong error: {other:?}"),
        }
    }
}

//! End-to-end transcription against a real model and real speech.
//!
//! Ignored by default: it needs a downloaded model and macOS `say` to
//! synthesise the audio. Run it with
//!
//! ```text
//! cargo test --test live_transcribe --features stt-whisper -- --ignored --nocapture
//! ```
//!
//! This exists because every unit test in the crate passed while transcription
//! returned nothing at all. `set_detect_language(true)` makes whisper.cpp run
//! language detection and return *before* decoding, so the language came back
//! correct and the transcript was silently empty. Only running real audio
//! through a real model catches that class of bug.

#![cfg(feature = "stt-whisper")]

use aria_ai_core::audio::{wav, SincResampler, TARGET_RATE_HZ};
use aria_ai_core::engine::{AiConfig, AiEngine};
use aria_ai_core::models::DeviceCapability;
use aria_ai_core::stt::TranscribeRequest;
use aria_ai_core::types::Speaker;
use std::path::{Path, PathBuf};
use std::process::Command;

const MODEL: &str = "whisper-base-q5_1";
const SPOKEN: &str = "The quick brown fox jumps over the lazy dog.";

/// Synthesise `SPOKEN` as a WAV at `rate`, mirroring a call recording.
fn synthesise(dir: &Path, rate: u32) -> Option<PathBuf> {
    let aiff = dir.join("speech.aiff");
    let wav_path = dir.join(format!("speech-{rate}.wav"));

    let said = Command::new("say")
        .args(["-o", aiff.to_str()?, SPOKEN])
        .status()
        .ok()?;
    if !said.success() {
        return None;
    }
    let converted = Command::new("afconvert")
        .args([
            "-f",
            "WAVE",
            "-d",
            &format!("LEI16@{rate}"),
            "-c",
            "1",
            aiff.to_str()?,
            wav_path.to_str()?,
        ])
        .status()
        .ok()?;
    converted.success().then_some(wav_path)
}

fn engine(root: &Path) -> AiEngine {
    AiEngine::new(AiConfig::new(root, DeviceCapability::desktop_default()))
        .expect("engine should start against a temp root")
}

/// The regression this file exists for: real speech in, real words out.
#[test]
#[ignore = "needs a downloaded model and macOS `say`"]
fn real_speech_produces_a_non_empty_transcript() {
    let root = std::env::var("ARIA_AI_ROOT")
        .map(PathBuf::from)
        .expect("set ARIA_AI_ROOT to a storage root with the model installed");
    let tmp = tempfile::tempdir().expect("temp dir");

    let Some(path) = synthesise(tmp.path(), 16_000) else {
        eprintln!("skipping: `say`/`afconvert` unavailable");
        return;
    };

    let e = engine(&root);
    let transcriber = e
        .transcriber(MODEL)
        .unwrap_or_else(|_| panic!("download {MODEL} into ARIA_AI_ROOT first"));

    let (samples, rate) = wav::read_mono_i16(&path).expect("reads back the wav it just wrote");
    let pcm = SincResampler::resample_all(rate, TARGET_RATE_HZ, &samples).expect("resamples");

    let out = transcriber
        .transcribe(&TranscribeRequest {
            samples: &pcm,
            sample_rate: TARGET_RATE_HZ,
            speaker: Speaker::Remote,
            // None means auto-detect — the exact path that was broken.
            language: None,
        })
        .expect("transcription should succeed");

    assert!(
        !out.utterances.is_empty(),
        "auto-detect produced no utterances at all — whisper detected the \
         language and returned before decoding"
    );

    let text = out
        .utterances
        .iter()
        .map(|u| u.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    // Not an exact match: a small model is allowed to mishear a word. These
    // are the content words it must not lose entirely.
    for word in ["quick", "brown", "fox", "lazy", "dog"] {
        assert!(
            text.contains(word),
            "expected {word:?} somewhere in {text:?}"
        );
    }
    assert_eq!(
        out.language, "en",
        "English speech should detect as English"
    );
}

/// A call is carried at 8 kHz, so the narrowband path is the one that actually
/// ships. It must survive the upsample to whisper's rate.
#[test]
#[ignore = "needs a downloaded model and macOS `say`"]
fn narrowband_call_audio_still_transcribes() {
    let root = std::env::var("ARIA_AI_ROOT")
        .map(PathBuf::from)
        .expect("set ARIA_AI_ROOT to a storage root with the model installed");
    let tmp = tempfile::tempdir().expect("temp dir");

    let Some(path) = synthesise(tmp.path(), 8_000) else {
        eprintln!("skipping: `say`/`afconvert` unavailable");
        return;
    };

    let e = engine(&root);
    let transcriber = e
        .transcriber(MODEL)
        .unwrap_or_else(|_| panic!("download {MODEL} into ARIA_AI_ROOT first"));

    let (samples, rate) = wav::read_mono_i16(&path).expect("reads back the wav");
    assert_eq!(rate, 8_000, "the fixture should be narrowband");
    let pcm = SincResampler::resample_all(rate, TARGET_RATE_HZ, &samples).expect("resamples");

    let out = transcriber
        .transcribe(&TranscribeRequest {
            samples: &pcm,
            sample_rate: TARGET_RATE_HZ,
            speaker: Speaker::Remote,
            language: None,
        })
        .expect("transcription should succeed");

    let text = out
        .utterances
        .iter()
        .map(|u| u.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        text.contains("fox") && text.contains("dog"),
        "narrowband transcript lost its content words: {text:?}"
    );
}

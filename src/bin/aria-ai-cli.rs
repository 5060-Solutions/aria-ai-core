//! `aria-ai-cli` — a desktop harness for driving `aria-ai-core` by hand.
//!
//! This exists so the pipeline can be exercised without a phone: download and
//! verify weights, run a real WAV through STT, summarise it, and inspect the
//! encrypted insight store. It is a development and benchmarking tool, not a
//! shipped product surface.
//!
//! ```text
//! aria-ai-cli models                      # catalogue, with what this box can run
//! aria-ai-cli download <model-id>         # fetch + verify (resumable)
//! aria-ai-cli verify <model-id>           # re-hash what is on disk
//! aria-ai-cli delete <model-id>
//! aria-ai-cli transcribe <file.wav> --stt <model-id>
//! aria-ai-cli summarize <file.wav> --stt <id> --llm <id>
//! aria-ai-cli insights list|show|rm|clear
//! ```
//!
//! Transcription and summarisation need the inference features:
//!
//! ```text
//! cargo run --features inference --bin aria-ai-cli -- transcribe call.wav --stt whisper-base-q5_1
//! ```
//!
//! Without them the commands still parse and tell you which feature is
//! missing, rather than pretending the model is unavailable.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use aria_ai_core::engine::{AiConfig, AiEngine};
use aria_ai_core::error::AiError;
use aria_ai_core::models::capability::{DeviceCapability, Platform};
use aria_ai_core::models::catalog::ModelKind;
use aria_ai_core::models::download::{DownloadProgress, DownloadState};
use aria_ai_core::llm::pipeline::SummaryConfig;
use aria_ai_core::llm::prompt::CallContext;
use aria_ai_core::session::SessionConfig;
use aria_ai_core::types::{CallInsight, Speaker};

#[cfg(feature = "insights")]
use aria_ai_core::insights::{InsightStore, KEY_LEN};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help" | "help") {
        usage();
        return ExitCode::SUCCESS;
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    eprintln!(
        "aria-ai-cli — drive aria-ai-core from a desktop shell

USAGE:
  aria-ai-cli <command> [args]

COMMANDS:
  models                          list the catalogue and what this machine can run
  download <model-id>             download and verify (resumes an interrupted one)
  verify <model-id>               re-hash the installed file(s) against the catalogue
  delete <model-id>               remove an installed model
  transcribe <file.wav>           transcribe a WAV  [--stt <id>] [--lang <code>]
  summarize <file.wav>            transcribe then summarise  [--stt <id>] [--llm <id>]
  insights list                   list stored insights (metadata only)
  insights show <call-id>         decrypt and print one insight
  insights rm <call-id>           delete one
  insights clear                  delete all

OPTIONS:
  --root <dir>    storage root (default: ./.aria-ai)
  --json          machine-readable output where it makes sense

ENVIRONMENT:
  ARIA_AI_ROOT      same as --root
  ARIA_AI_KEY_HEX   64 hex chars: the insight-store key. On a real device this
                    comes from the Keychain/Keystore; here you supply it. Set it
                    to the same value between runs or your rows are unreadable."
    );
}

fn run(args: &[String]) -> Result<(), AiError> {
    let root = storage_root(args);
    let json = args.iter().any(|a| a == "--json");

    match args[0].as_str() {
        "models" => cmd_models(&root, json),
        "download" => cmd_download(&root, arg_at(args, 1, "model-id")?),
        "verify" => cmd_verify(&root, arg_at(args, 1, "model-id")?),
        "delete" => cmd_delete(&root, arg_at(args, 1, "model-id")?),
        "transcribe" => cmd_transcribe(&root, args, false, json),
        "summarize" | "summarise" => cmd_transcribe(&root, args, true, json),
        "insights" => cmd_insights(&root, args, json),
        other => Err(AiError::Config(format!(
            "unknown command {other:?} — try `aria-ai-cli help`"
        ))),
    }
}

// ── shared helpers ──────────────────────────────────────────────────────────

fn arg_at<'a>(args: &'a [String], i: usize, what: &str) -> Result<&'a str, AiError> {
    args.get(i)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .ok_or_else(|| AiError::Config(format!("expected a {what}")))
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let pos = args.iter().position(|a| a == name)?;
    args.get(pos + 1).map(String::as_str)
}

fn storage_root(args: &[String]) -> PathBuf {
    flag(args, "--root")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("ARIA_AI_ROOT").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(".aria-ai"))
}

/// Describe the machine we are actually on, so gating behaves like a device.
fn this_machine() -> DeviceCapability {
    let cores = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
    // No portable free-RAM probe without another dependency; assume a
    // developer box is roomy and say so rather than inventing a number.
    let total_ram_mb = 16_384;
    let platform = if cfg!(target_os = "macos") {
        Platform::MacOs
    } else if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "linux") {
        Platform::Linux
    } else {
        Platform::Other
    };
    DeviceCapability {
        platform,
        total_ram_mb,
        available_ram_mb: total_ram_mb / 2,
        cpu_cores: u32::try_from(cores).unwrap_or(4),
        is_64bit: cfg!(target_pointer_width = "64"),
        has_gpu_backend: cfg!(any(target_os = "macos", target_os = "windows")),
    }
}

fn engine(root: &Path) -> Result<AiEngine, AiError> {
    AiEngine::new(AiConfig::new(root, this_machine()))
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    #[allow(clippy::cast_precision_loss)]
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

// ── commands ────────────────────────────────────────────────────────────────

fn cmd_models(root: &Path, json: bool) -> Result<(), AiError> {
    let e = engine(root)?;
    let availability = e.availability();

    if json {
        let out = serde_json::to_string_pretty(&availability)
            .map_err(|err| AiError::Internal(err.to_string()))?;
        println!("{out}");
        return Ok(());
    }

    let installed: Vec<String> = e.installed_models().into_iter().map(|m| m.id).collect();
    println!(
        "{:<32} {:<22} {:>9}  {:<9} NOTE",
        "ID", "NAME", "SIZE", "STATE"
    );
    for a in &availability {
        let m = &a.model;
        let state = if installed.contains(&m.id) {
            "installed"
        } else if a.available {
            "available"
        } else {
            "blocked"
        };
        let note = a.reason.clone().unwrap_or_else(|| {
            if m.is_multi_file() {
                format!("{} files", m.files.len())
            } else {
                String::new()
            }
        });
        println!(
            "{:<32} {:<22} {:>9}  {:<9} {}",
            m.id,
            m.display_name,
            human_bytes(m.size_bytes),
            state,
            note
        );
    }
    println!("\nstorage: {} in {}", human_bytes(e.storage_bytes()), root.display());
    for kind in [ModelKind::Stt, ModelKind::Llm] {
        if let Some(r) = e.recommended(kind) {
            println!("recommended {kind:?}: {}", r.id);
        }
    }
    Ok(())
}

fn cmd_download(root: &Path, model_id: &str) -> Result<(), AiError> {
    let e = engine(root)?;
    let descriptor = e.descriptor(model_id)?;
    println!(
        "{} — {} ({})",
        descriptor.id,
        descriptor.display_name,
        human_bytes(descriptor.size_bytes)
    );
    println!("license: {}", descriptor.attribution);

    e.start_download(model_id)?;

    let mut last_line = String::new();
    loop {
        let Some(p) = e.download_progress(model_id) else {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        };
        let line = render_progress(&p);
        if line != last_line {
            eprint!("\r{line}    ");
            last_line = line;
        }
        if p.is_terminal() {
            eprintln!();
            return match p.state {
                DownloadState::Completed => {
                    let installed = e.installed_models();
                    if let Some(m) = installed.iter().find(|m| m.id == model_id) {
                        println!("installed at {}", m.path);
                    }
                    println!("verified against the catalogue digest");
                    Ok(())
                }
                DownloadState::Cancelled => Err(AiError::DownloadCancelled),
                DownloadState::Failed { reason } => Err(AiError::Download(reason)),
                other => Err(AiError::Internal(format!("unexpected state {other:?}"))),
            };
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn render_progress(p: &DownloadProgress) -> String {
    let pct = p.fraction() * 100.0;
    let rate = if p.bytes_per_sec > 1.0 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let r = p.bytes_per_sec as u64;
        format!(" {}/s", human_bytes(r))
    } else {
        String::new()
    };
    let eta = p.eta_secs.map_or_else(String::new, |s| format!(" eta {s}s"));
    format!(
        "{:?}: {:.1}% ({} / {}){rate}{eta}",
        p.state,
        pct,
        human_bytes(p.downloaded_bytes),
        human_bytes(p.total_bytes)
    )
}

fn cmd_verify(root: &Path, model_id: &str) -> Result<(), AiError> {
    let e = engine(root)?;
    e.verify_model(model_id)?;
    println!("{model_id}: digest matches the catalogue");
    Ok(())
}

fn cmd_delete(root: &Path, model_id: &str) -> Result<(), AiError> {
    let e = engine(root)?;
    let freed = e.delete_model(model_id)?;
    println!("{model_id}: removed, {} reclaimed", human_bytes(freed));
    Ok(())
}

fn cmd_transcribe(
    root: &Path,
    args: &[String],
    summarize: bool,
    json: bool,
) -> Result<(), AiError> {
    let path = PathBuf::from(arg_at(args, 1, "path to a .wav file")?);
    let e = engine(root)?;

    let stt_id = flag(args, "--stt")
        .map(str::to_string)
        .or_else(|| e.recommended(ModelKind::Stt).map(|m| m.id.clone()))
        .ok_or_else(|| AiError::Config("no STT model — pass --stt <id>".into()))?;

    let (samples, rate) = read_wav_mono_i16(&path)?;
    #[allow(clippy::cast_precision_loss)]
    let seconds = samples.len() as f64 / f64::from(rate);
    eprintln!(
        "{}: {} samples @ {rate} Hz ({seconds:.1}s)",
        path.display(),
        samples.len()
    );

    let transcriber = e.transcriber(&stt_id)?;

    let mut config = SessionConfig::new(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("cli-call"),
    );
    if let Some(lang) = flag(args, "--lang") {
        config = config.with_language(lang);
    }
    let session = e.begin_call(config)?;

    // A mono file has no leg separation to recover, so it is fed as the remote
    // side and labelled that way rather than guessing at a speaker split.
    session.push_remote(&samples, rate)?;
    session.pump()?;
    session.stop()?;

    let transcript = session.transcribe(transcriber.as_ref())?;
    if !json {
        println!("\n── transcript ({}) ──", transcript.language);
        for s in &transcript.segments {
            let who = match s.speaker {
                Speaker::Local => "you",
                Speaker::Remote => "them",
            };
            println!(
                "[{:>7} → {:>7}] {who:>4}: {}",
                fmt_ms(s.start_ms),
                fmt_ms(s.end_ms),
                s.text
            );
        }
    }

    if summarize {
        let llm_id = flag(args, "--llm")
            .map(str::to_string)
            .or_else(|| e.recommended(ModelKind::Llm).map(|m| m.id.clone()))
            .ok_or_else(|| AiError::Config("no LLM model — pass --llm <id>".into()))?;
        let summarizer = e.summarizer(&llm_id)?;
        let ctx = CallContext {
            remote_label: flag(args, "--remote").unwrap_or("unknown").to_string(),
            // The crate carries no date library on purpose; the host formats
            // dates, and here the host is a shell.
            date_label: flag(args, "--date").unwrap_or("an unspecified date").to_string(),
            duration_secs: transcript.duration_ms / 1000,
        };
        let summary =
            session.summarize(summarizer.as_ref(), &ctx, &SummaryConfig::default())?;
        if !json {
            println!("\n── summary ──");
            if summary.degraded {
                println!("(degraded: structured parsing failed, this is a text fallback)");
            }
            println!("{}", summary.headline);
            println!("{}", summary.abstract_text);
            for p in &summary.key_points {
                println!("  • {p}");
            }
            for a in &summary.action_items {
                println!("  → {} [{:?}]", a.text, a.owner);
            }
        }
    }

    let insight = session.insight();
    if json {
        let out = serde_json::to_string_pretty(&insight)
            .map_err(|err| AiError::Internal(err.to_string()))?;
        println!("{out}");
    }

    store_insight(root, &insight);
    Ok(())
}

fn fmt_ms(ms: u32) -> String {
    format!("{}:{:05.2}", ms / 60_000, f64::from(ms % 60_000) / 1000.0)
}

// ── insight store ───────────────────────────────────────────────────────────

#[cfg(feature = "insights")]
fn insight_key() -> Result<[u8; KEY_LEN], AiError> {
    let hex = std::env::var("ARIA_AI_KEY_HEX").map_err(|_| {
        AiError::Config(
            "set ARIA_AI_KEY_HEX to 64 hex characters (the key a device would \
             hold in its Keychain/Keystore)"
                .into(),
        )
    })?;
    let hex = hex.trim();
    if hex.len() != KEY_LEN * 2 {
        return Err(AiError::Config(format!(
            "ARIA_AI_KEY_HEX must be {} hex characters, got {}",
            KEY_LEN * 2,
            hex.len()
        )));
    }
    let mut key = [0_u8; KEY_LEN];
    for (i, byte) in key.iter_mut().enumerate() {
        let pair = hex
            .get(i * 2..i * 2 + 2)
            .ok_or_else(|| AiError::Config("ARIA_AI_KEY_HEX is malformed".into()))?;
        *byte = u8::from_str_radix(pair, 16)
            .map_err(|_| AiError::Config("ARIA_AI_KEY_HEX is not hex".into()))?;
    }
    Ok(key)
}

#[cfg(feature = "insights")]
fn open_insights(root: &Path) -> Result<InsightStore, AiError> {
    std::fs::create_dir_all(root).map_err(|e| AiError::io(root, &e))?;
    InsightStore::open(root.join("insights.db"), &insight_key()?)
}

/// Persist the insight, best effort.
///
/// The transcript has already been printed by the time this runs, so a storage
/// problem is worth reporting but not worth failing the command over.
#[cfg(feature = "insights")]
fn store_insight(root: &Path, insight: &CallInsight) {
    if std::env::var_os("ARIA_AI_KEY_HEX").is_none() {
        eprintln!("\n(not stored: ARIA_AI_KEY_HEX is unset)");
        return;
    }
    match open_insights(root).and_then(|s| s.put(insight)) {
        Ok(()) => eprintln!("\nstored insight {}", insight.call_id),
        Err(e) => eprintln!("\nwarning: could not store insight: {e}"),
    }
}

#[cfg(not(feature = "insights"))]
fn store_insight(_root: &Path, _insight: &CallInsight) {
    eprintln!("\n(not stored: built without the `insights` feature)");
}

#[cfg(feature = "insights")]
fn cmd_insights(root: &Path, args: &[String], json: bool) -> Result<(), AiError> {
    let sub = args.get(1).map_or("list", String::as_str);
    let store = open_insights(root)?;
    match sub {
        "list" => {
            let rows = store.list(100, 0)?;
            if json {
                // InsightSummaryRow is deliberately not Serialize (it is a UI
                // record), so emit the fields explicitly.
                let v: Vec<serde_json::Value> = rows
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "call_id": r.call_id,
                            "created_at": r.created_at,
                            "duration_secs": r.duration_secs,
                            "status": r.status,
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&v)
                        .map_err(|e| AiError::Internal(e.to_string()))?
                );
            } else {
                println!("{:<28} {:>12} {:>9}  STATUS", "CALL", "CREATED", "DURATION");
                for r in &rows {
                    println!(
                        "{:<28} {:>12} {:>8}s  {}",
                        r.call_id, r.created_at, r.duration_secs, r.status
                    );
                }
                println!("\n{} stored", store.count()?);
            }
            Ok(())
        }
        "show" => {
            let id = arg_at(args, 2, "call-id")?;
            let insight = store
                .get(id)?
                .ok_or_else(|| AiError::Config(format!("no insight for {id}")))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&insight)
                    .map_err(|e| AiError::Internal(e.to_string()))?
            );
            Ok(())
        }
        "rm" => {
            let id = arg_at(args, 2, "call-id")?;
            if store.delete(id)? {
                println!("deleted {id}");
            } else {
                println!("no insight for {id}");
            }
            Ok(())
        }
        "clear" => {
            let n = store.clear()?;
            println!("deleted {n} insight(s)");
            Ok(())
        }
        other => Err(AiError::Config(format!("unknown insights subcommand {other:?}"))),
    }
}

#[cfg(not(feature = "insights"))]
fn cmd_insights(_root: &Path, _args: &[String], _json: bool) -> Result<(), AiError> {
    Err(AiError::Config(
        "built without the `insights` feature — rebuild with --features insights".into(),
    ))
}

// ── WAV reading ─────────────────────────────────────────────────────────────

/// Minimal RIFF/WAVE reader for 16-bit PCM, mixing to mono.
///
/// Deliberately hand-rolled: the crate ships no audio-file dependency, and a
/// development harness should not add one to the library's tree.
fn read_wav_mono_i16(path: &Path) -> Result<(Vec<i16>, u32), AiError> {
    let bytes = std::fs::read(path).map_err(|e| AiError::io(path, &e))?;
    let bad = |what: &str| AiError::Config(format!("{}: {what}", path.display()));

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }

    let u16_at = |o: usize| -> u16 { u16::from_le_bytes([bytes[o], bytes[o + 1]]) };
    let u32_at = |o: usize| -> u32 {
        u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
    };

    let mut pos = 12;
    let mut channels = 0_u16;
    let mut rate = 0_u32;
    let mut bits = 0_u16;
    let mut format = 0_u16;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(pos + 4) as usize;
        let body = pos + 8;
        if body + size > bytes.len() {
            // Some encoders over-declare the final chunk; clamp rather than fail.
            if id == b"data" {
                let samples = decode_pcm(&bytes[body..], bits, channels, &bad)?;
                return Ok((samples, rate));
            }
            break;
        }

        match id {
            b"fmt " if size >= 16 => {
                format = u16_at(body);
                channels = u16_at(body + 2);
                rate = u32_at(body + 4);
                bits = u16_at(body + 14);
            }
            b"data" => {
                if format != 1 {
                    return Err(bad(&format!(
                        "only uncompressed PCM is supported (format {format})"
                    )));
                }
                let samples = decode_pcm(&bytes[body..body + size], bits, channels, &bad)?;
                return Ok((samples, rate));
            }
            _ => {}
        }
        // Chunks are word-aligned.
        pos = body + size + (size & 1);
    }
    Err(bad("no data chunk"))
}

fn decode_pcm(
    data: &[u8],
    bits: u16,
    channels: u16,
    bad: &dyn Fn(&str) -> AiError,
) -> Result<Vec<i16>, AiError> {
    if bits != 16 {
        return Err(bad(&format!("only 16-bit PCM is supported (got {bits})")));
    }
    let channels = usize::from(channels.max(1));
    // Small by construction (a WAV channel count), so this cannot lose data.
    let divisor = i32::try_from(channels).unwrap_or(1);
    let frames = data.len() / 2 / channels;
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        // Average the channels so a stereo recording does not come through at
        // double rate or silently lose a side.
        let mut acc = 0_i32;
        for c in 0..channels {
            let o = (f * channels + c) * 2;
            acc += i32::from(i16::from_le_bytes([data[o], data[o + 1]]));
        }
        out.push(i16::try_from(acc / divisor).unwrap_or(0));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal PCM WAV. Test-only, so a length that does not fit a
    /// u32 is a broken test rather than a case to handle.
    fn wav(channels: u16, rate: u32, samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let data_len = u32::try_from(data.len()).expect("test data fits a u32");
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data_len).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16_u32.to_le_bytes());
        v.extend_from_slice(&1_u16.to_le_bytes()); // PCM
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
        v.extend_from_slice(&(channels * 2).to_le_bytes());
        v.extend_from_slice(&16_u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        v.extend_from_slice(&data);
        v
    }

    fn write_temp(bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("a.wav");
        std::fs::write(&p, bytes).expect("write");
        (dir, p)
    }

    #[test]
    fn a_mono_wav_reads_back_sample_for_sample() {
        let samples = [0_i16, 1000, -1000, 32767, -32768];
        let (_d, p) = write_temp(&wav(1, 8000, &samples));
        let (got, rate) = read_wav_mono_i16(&p).expect("read");
        assert_eq!(rate, 8000);
        assert_eq!(got, samples);
    }

    #[test]
    fn a_stereo_wav_is_averaged_to_mono_not_interleaved() {
        // L/R pairs: (100,300) -> 200, (0,0) -> 0, (-100,-300) -> -200
        let interleaved = [100_i16, 300, 0, 0, -100, -300];
        let (_d, p) = write_temp(&wav(2, 16_000, &interleaved));
        let (got, rate) = read_wav_mono_i16(&p).expect("read");
        assert_eq!(rate, 16_000);
        assert_eq!(got, vec![200, 0, -200], "frames, not samples");
    }

    #[test]
    fn a_wav_with_an_extra_chunk_before_data_is_still_read() {
        // Real recorders emit LIST/INFO between fmt and data.
        let mut v = wav(1, 8000, &[7_i16, 8, 9]);
        let data_at = v
            .windows(4)
            .position(|w| w == b"data")
            .expect("data chunk");
        let extra: Vec<u8> = b"LIST\x04\x00\x00\x00INFO".to_vec();
        v.splice(data_at..data_at, extra.iter().copied());
        let riff_len = u32::try_from(v.len() - 8).expect("fits a u32");
        v[4..8].copy_from_slice(&riff_len.to_le_bytes());

        let (_d, p) = write_temp(&v);
        let (got, _) = read_wav_mono_i16(&p).expect("read");
        assert_eq!(got, vec![7, 8, 9]);
    }

    #[test]
    fn non_wav_and_unsupported_encodings_are_rejected_clearly() {
        let (_d, p) = write_temp(b"this is not a wav");
        assert!(read_wav_mono_i16(&p).is_err());

        // 8-bit PCM: valid RIFF, unsupported depth.
        let mut v = wav(1, 8000, &[1_i16]);
        let bits_at = v.windows(4).position(|w| w == b"WAVE").expect("wave") + 4 + 8 + 14;
        v[bits_at..bits_at + 2].copy_from_slice(&8_u16.to_le_bytes());
        let (_d2, p2) = write_temp(&v);
        let err = read_wav_mono_i16(&p2).expect_err("should reject");
        assert!(format!("{err}").contains("16-bit"), "{err}");
    }

    #[test]
    fn flags_and_positional_arguments_parse_independently() {
        let args: Vec<String> = ["transcribe", "call.wav", "--stt", "whisper-base-q5_1", "--json"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        assert_eq!(arg_at(&args, 1, "path").expect("path"), "call.wav");
        assert_eq!(flag(&args, "--stt"), Some("whisper-base-q5_1"));
        assert_eq!(flag(&args, "--llm"), None);
    }

    #[test]
    fn a_missing_positional_is_an_error_not_a_flag() {
        let args: Vec<String> = ["download", "--json"].iter().map(|s| (*s).to_string()).collect();
        assert!(arg_at(&args, 1, "model-id").is_err());
    }

    #[test]
    fn byte_sizes_render_the_way_a_human_reads_them() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KB");
        assert_eq!(human_bytes(59_707_625), "56.9 MB");
    }
}

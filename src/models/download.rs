//! Resumable, cancellable, integrity-checked model download (design spec §6.3).
//!
//! The transport is behind [`ModelSource`] so the whole state machine —
//! resume, server-ignored-Range recovery, cancellation, hash verification,
//! atomic install — is unit-testable without a network. The real HTTPS
//! implementation lives in [`crate::models::http`] behind the `net` feature.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{AiError, Result};
use crate::models::catalog::{check_host, ModelDescriptor};
use crate::models::store::{sha256_file, ModelStore};
use crate::util::to_hex;

/// Bytes pulled from the transport per progress tick.
const READ_CHUNK: usize = 128 * 1024;

/// One open response body.
pub struct SourceStream {
    /// Total size of the *whole* resource, if the server said.
    pub total_bytes: Option<u64>,
    /// Byte offset the body actually starts at. A server that ignored our
    /// `Range` header reports 0 here and the download restarts cleanly.
    pub start_offset: u64,
    /// The body.
    pub reader: Box<dyn Read + Send>,
}

/// Where model bytes come from.
///
/// Implementors must reject non-HTTPS and off-allowlist URLs; the downloader
/// checks too, so this is defence in depth.
pub trait ModelSource: std::fmt::Debug + Send + Sync {
    /// Open `url`, requesting bytes from `offset` onwards.
    ///
    /// # Errors
    /// [`AiError::Download`] for transport failures,
    /// [`AiError::HostNotAllowed`] for a URL off the allowlist.
    fn open(&self, url: &str, offset: u64) -> Result<SourceStream>;
}

/// Cooperative cancellation, checked once per chunk.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A fresh, un-cancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the download to stop. The `.part` file is kept so it can resume.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Has cancellation been requested?
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Lifecycle of one download.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum DownloadState {
    /// Accepted, not started.
    Queued,
    /// Bytes moving.
    Running,
    /// All bytes in, hashing.
    Verifying,
    /// Installed.
    Completed,
    /// Stopped by the user; `.part` retained.
    Cancelled,
    /// Gave up.
    Failed { reason: String },
}

/// Polled progress snapshot (design spec §2.4 — polling, not callbacks).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct DownloadProgress {
    pub model_id: String,
    pub downloaded_bytes: u64,
    /// From the server when known, otherwise the catalogue's figure.
    pub total_bytes: u64,
    pub bytes_per_sec: f64,
    /// `None` while the rate is still unknown.
    pub eta_secs: Option<u64>,
    pub state: DownloadState,
}

impl DownloadProgress {
    /// Completion in `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fraction(&self) -> f32 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.downloaded_bytes as f32 / self.total_bytes as f32).clamp(0.0, 1.0)
    }

    /// Is this download over, one way or another?
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            DownloadState::Completed | DownloadState::Cancelled | DownloadState::Failed { .. }
        )
    }
}

/// Drives one download at a time against a [`ModelSource`].
#[derive(Debug, Clone)]
pub struct Downloader {
    store: ModelStore,
    source: Arc<dyn ModelSource>,
}

impl Downloader {
    /// Build a downloader over a store and a transport.
    #[must_use]
    pub fn new(store: ModelStore, source: Arc<dyn ModelSource>) -> Self {
        Self { store, source }
    }

    /// The store this downloader installs into.
    #[must_use]
    pub const fn store(&self) -> &ModelStore {
        &self.store
    }

    /// Download, verify and atomically install `model`.
    ///
    /// Resumes from an existing `.part` file when there is one. `on_progress`
    /// is called at least once per chunk; it must not block for long.
    ///
    /// # Errors
    /// - [`AiError::HostNotAllowed`] — URL off the compile-time allowlist.
    /// - [`AiError::Config`] — the catalogue entry still has a placeholder digest.
    /// - [`AiError::InsufficientSpace`] — free-space preflight failed.
    /// - [`AiError::DownloadCancelled`] — cancelled; `.part` kept for resume.
    /// - [`AiError::IntegrityMismatch`] — hash mismatch; `.part` deleted.
    /// - [`AiError::Download`] / [`AiError::Io`] — transport or disk failure.
    pub fn download(
        &self,
        model: &ModelDescriptor,
        cancel: &CancelToken,
        on_progress: &mut dyn FnMut(&DownloadProgress),
    ) -> Result<std::path::PathBuf> {
        if !model.has_real_checksum() {
            return Err(AiError::Config(format!(
                "model {} has a placeholder checksum; refusing to install an unverified file",
                model.id
            )));
        }
        self.store.ensure_dirs()?;
        self.store.preflight(model)?;

        if model.is_multi_file() {
            self.download_multi(model, cancel, on_progress)
        } else {
            self.download_single(model, cancel, on_progress)
        }
    }

    /// One blob straight into `models/<file_name>`.
    fn download_single(
        &self,
        model: &ModelDescriptor,
        cancel: &CancelToken,
        on_progress: &mut dyn FnMut(&DownloadProgress),
    ) -> Result<std::path::PathBuf> {
        let final_path = self.store.path_for(model);
        if final_path.is_file() && self.store.verify(model).is_ok() {
            on_progress(&progress_of(
                model,
                model.size_bytes,
                model.size_bytes,
                0.0,
                DownloadState::Completed,
            ));
            return Ok(final_path);
        }

        let spec = FetchSpec {
            url: &model.url,
            size_bytes: model.size_bytes,
            sha256: &model.sha256,
            label: &model.id,
            part_path: self.store.part_path_for(model),
            final_path: final_path.clone(),
        };
        self.fetch(&spec, model, 0, model.size_bytes, cancel, on_progress)?;

        on_progress(&progress_of(
            model,
            model.size_bytes,
            model.size_bytes,
            0.0,
            DownloadState::Completed,
        ));
        Ok(final_path)
    }

    /// A directory of members, each fetched and verified on its own.
    ///
    /// Progress is reported against the whole model, so a caller driving a
    /// single progress bar sees one monotonic run across all four Parakeet
    /// files rather than four bars restarting from zero.
    fn download_multi(
        &self,
        model: &ModelDescriptor,
        cancel: &CancelToken,
        on_progress: &mut dyn FnMut(&DownloadProgress),
    ) -> Result<std::path::PathBuf> {
        let dir = self.store.path_for(model);
        if self.store.is_installed(model) && self.store.verify(model).is_ok() {
            on_progress(&progress_of(
                model,
                model.size_bytes,
                model.size_bytes,
                0.0,
                DownloadState::Completed,
            ));
            return Ok(dir);
        }
        fs::create_dir_all(&dir).map_err(|e| AiError::io(&dir, &e))?;

        let total = model.size_bytes;
        let mut base = 0_u64;
        for file in &model.files {
            let member_path = self.store.member_path(model, file);

            // Keep a member that is already there and correct: re-downloading
            // 650 MB because a 9 KB sibling failed would be indefensible.
            if member_path.is_file()
                && sha256_file(&member_path)
                    .is_ok_and(|actual| actual.eq_ignore_ascii_case(&file.sha256))
            {
                base = base.saturating_add(file.size_bytes);
                on_progress(&progress_of(model, base, total, 0.0, DownloadState::Running));
                continue;
            }

            let spec = FetchSpec {
                url: &file.url,
                size_bytes: file.size_bytes,
                sha256: &file.sha256,
                label: &format!("{}/{}", model.id, file.file_name),
                part_path: self.store.member_part_path(model, file),
                final_path: member_path,
            };
            self.fetch(&spec, model, base, total, cancel, on_progress)?;
            base = base.saturating_add(file.size_bytes);
        }

        on_progress(&progress_of(model, total, total, 0.0, DownloadState::Completed));
        Ok(dir)
    }

    /// Fetch one file: resume, stream, hash, size-check, atomic install.
    ///
    /// `base_bytes` is how much of the overall model was already done before
    /// this file, and `total_bytes` the whole model, so the progress a caller
    /// sees is always about the model rather than the current member.
    fn fetch(
        &self,
        spec: &FetchSpec<'_>,
        model: &ModelDescriptor,
        base_bytes: u64,
        total_bytes: u64,
        cancel: &CancelToken,
        on_progress: &mut dyn FnMut(&DownloadProgress),
    ) -> Result<()> {
        check_host(spec.url)?;

        let part_path = &spec.part_path;
        let mut downloaded = fs::metadata(part_path).map_or(0, |m| m.len());
        let mut hasher = Sha256::new();

        // Resuming means re-hashing what is already on disk: SHA-256 state
        // cannot be persisted across runs, and re-reading a partial file is
        // far cheaper than re-downloading it.
        if downloaded > 0 && hash_prefix(part_path, downloaded, &mut hasher).is_err() {
            // Unreadable partial: start over rather than install junk.
            hasher = Sha256::new();
            downloaded = 0;
            let _ = fs::remove_file(part_path);
        }

        on_progress(&progress_of(
            model,
            base_bytes.saturating_add(downloaded),
            total_bytes,
            0.0,
            DownloadState::Queued,
        ));

        let mut stream = self.source.open(spec.url, downloaded)?;

        // Server ignored Range and is sending from the top: discard what we
        // had rather than splicing two byte ranges together.
        if stream.start_offset == 0 && downloaded > 0 {
            hasher = Sha256::new();
            downloaded = 0;
        } else if stream.start_offset != downloaded {
            return Err(AiError::Download(format!(
                "server resumed at byte {} but {} bytes were already downloaded",
                stream.start_offset, downloaded
            )));
        }

        let mut file = open_part(part_path, downloaded)?;
        let started = Instant::now();
        let start_bytes = downloaded;
        let mut buf = vec![0_u8; READ_CHUNK];

        loop {
            if cancel.is_cancelled() {
                let _ = file.flush();
                on_progress(&progress_of(
                    model,
                    base_bytes.saturating_add(downloaded),
                    total_bytes,
                    0.0,
                    DownloadState::Cancelled,
                ));
                return Err(AiError::DownloadCancelled);
            }
            let n = stream
                .reader
                .read(&mut buf)
                .map_err(|e| AiError::Download(e.to_string()))?;
            if n == 0 {
                break;
            }
            let chunk = buf.get(..n).unwrap_or(&[]);
            file.write_all(chunk).map_err(|e| AiError::io(part_path, &e))?;
            hasher.update(chunk);
            downloaded = downloaded.saturating_add(n as u64);

            let rate = rate_of(downloaded - start_bytes, started);
            on_progress(&progress_of(
                model,
                base_bytes.saturating_add(downloaded),
                total_bytes,
                rate,
                DownloadState::Running,
            ));
        }

        file.flush().map_err(|e| AiError::io(part_path, &e))?;
        file.sync_all().map_err(|e| AiError::io(part_path, &e))?;
        drop(file);

        on_progress(&progress_of(
            model,
            base_bytes.saturating_add(downloaded),
            total_bytes,
            0.0,
            DownloadState::Verifying,
        ));

        if downloaded != spec.size_bytes {
            let _ = fs::remove_file(part_path);
            return Err(AiError::Download(format!(
                "expected {} bytes, received {downloaded}",
                spec.size_bytes
            )));
        }

        let actual = to_hex(&hasher.finalize());
        if !actual.eq_ignore_ascii_case(spec.sha256) {
            // The catalogue hash is compiled into a signed binary, so a
            // mismatch means the CDN is wrong or hostile. Do not retry.
            let _ = fs::remove_file(part_path);
            return Err(AiError::IntegrityMismatch {
                id: spec.label.to_string(),
                expected: spec.sha256.to_ascii_lowercase(),
                actual,
            });
        }

        // Atomic install: never leave a half file where the loader looks.
        fs::rename(part_path, &spec.final_path)
            .map_err(|e| AiError::io(&spec.final_path, &e))?;
        Ok(())
    }
}

/// One file to fetch, independent of whether it is a whole model or a member.
struct FetchSpec<'a> {
    url: &'a str,
    size_bytes: u64,
    sha256: &'a str,
    /// Identifies the file in an integrity error.
    label: &'a str,
    part_path: std::path::PathBuf,
    final_path: std::path::PathBuf,
}

fn progress_of(
    model: &ModelDescriptor,
    downloaded: u64,
    total: u64,
    rate: f64,
    state: DownloadState,
) -> DownloadProgress {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let eta = if rate > 1.0 && total > downloaded {
        Some(((total - downloaded) as f64 / rate) as u64)
    } else {
        None
    };
    DownloadProgress {
        model_id: model.id.clone(),
        downloaded_bytes: downloaded,
        total_bytes: total,
        bytes_per_sec: rate,
        eta_secs: eta,
        state,
    }
}

#[allow(clippy::cast_precision_loss)]
fn rate_of(bytes: u64, since: Instant) -> f64 {
    let secs = since.elapsed().as_secs_f64();
    if secs <= f64::EPSILON {
        0.0
    } else {
        bytes as f64 / secs
    }
}

fn open_part(path: &std::path::Path, offset: u64) -> Result<File> {
    if offset == 0 {
        let f = File::create(path).map_err(|e| AiError::io(path, &e))?;
        return Ok(f);
    }
    let mut f = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| AiError::io(path, &e))?;
    f.seek(SeekFrom::Start(offset))
        .map_err(|e| AiError::io(path, &e))?;
    // Truncate anything past the resume point so a short write cannot leave
    // stale bytes beyond the hashed prefix.
    f.set_len(offset).map_err(|e| AiError::io(path, &e))?;
    f.seek(SeekFrom::Start(offset))
        .map_err(|e| AiError::io(path, &e))?;
    Ok(f)
}

fn hash_prefix(path: &std::path::Path, len: u64, hasher: &mut Sha256) -> Result<()> {
    let mut file = File::open(path).map_err(|e| AiError::io(path, &e))?;
    let mut remaining = len;
    let mut buf = vec![0_u8; READ_CHUNK];
    while remaining > 0 {
        let want = usize::try_from(remaining).unwrap_or(READ_CHUNK).min(READ_CHUNK);
        let n = file
            .read(buf.get_mut(..want).unwrap_or(&mut []))
            .map_err(|e| AiError::io(path, &e))?;
        if n == 0 {
            return Err(AiError::io_msg(path, "partial file shorter than expected"));
        }
        hasher.update(buf.get(..n).unwrap_or(&[]));
        remaining -= n as u64;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod fake {
    //! A [`ModelSource`] backed by an in-memory blob, with switches for every
    //! failure mode the real world produces.

    use super::{AiError, ModelSource, Result, SourceStream};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub enum Behaviour {
        /// Honour `Range`, serve to the end.
        #[default]
        Normal,
        /// Ignore `Range` and always serve from byte 0 (a real CDN habit).
        IgnoresRange,
        /// Serve `n` bytes then stop, simulating a dropped connection.
        TruncatesAfter(usize),
        /// Serve bytes that do not match the catalogue hash.
        ServesCorruptBytes,
        /// Refuse to open at all.
        Unreachable,
    }

    #[derive(Debug)]
    pub struct FakeSource {
        body: Vec<u8>,
        behaviour: Mutex<Behaviour>,
        pub opens: AtomicUsize,
        pub last_offset: AtomicUsize,
    }

    impl FakeSource {
        pub fn new(body: Vec<u8>) -> Arc<Self> {
            Arc::new(Self {
                body,
                behaviour: Mutex::new(Behaviour::Normal),
                opens: AtomicUsize::new(0),
                last_offset: AtomicUsize::new(0),
            })
        }

        pub fn set(&self, b: Behaviour) {
            *crate::util::lock(&self.behaviour) = b;
        }
    }

    impl ModelSource for FakeSource {
        fn open(&self, url: &str, offset: u64) -> Result<SourceStream> {
            crate::models::catalog::check_host(url)?;
            self.opens.fetch_add(1, Ordering::SeqCst);
            self.last_offset
                .store(usize::try_from(offset).unwrap_or(usize::MAX), Ordering::SeqCst);

            let behaviour = *crate::util::lock(&self.behaviour);
            if behaviour == Behaviour::Unreachable {
                return Err(AiError::Download("host unreachable".into()));
            }

            let mut body = self.body.clone();
            if behaviour == Behaviour::ServesCorruptBytes {
                if let Some(b) = body.first_mut() {
                    *b ^= 0xff;
                }
            }

            let start = if behaviour == Behaviour::IgnoresRange {
                0
            } else {
                usize::try_from(offset).unwrap_or(usize::MAX).min(body.len())
            };
            let mut slice = body.split_off(start);
            if let Behaviour::TruncatesAfter(n) = behaviour {
                slice.truncate(n);
            }

            Ok(SourceStream {
                total_bytes: Some(slice.len() as u64),
                start_offset: start as u64,
                reader: Box::new(std::io::Cursor::new(slice)),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{Behaviour, FakeSource};
    use super::*;
    use crate::models::capability::DeviceTier;
    use crate::models::catalog::{ModelFile, ModelKind, ModelRuntime};

    /// Serves a different body per URL, for multi-file models.
    #[derive(Debug)]
    struct MultiSource {
        bodies: std::collections::HashMap<String, Vec<u8>>,
        opens: std::sync::atomic::AtomicUsize,
    }

    impl MultiSource {
        fn new(bodies: Vec<(String, Vec<u8>)>) -> Arc<Self> {
            Arc::new(Self {
                bodies: bodies.into_iter().collect(),
                opens: std::sync::atomic::AtomicUsize::new(0),
            })
        }
    }

    impl ModelSource for MultiSource {
        fn open(&self, url: &str, offset: u64) -> Result<SourceStream> {
            crate::models::catalog::check_host(url)?;
            self.opens.fetch_add(1, Ordering::SeqCst);
            let body = self
                .bodies
                .get(url)
                .ok_or_else(|| AiError::Download(format!("no body for {url}")))?
                .clone();
            let start = usize::try_from(offset).unwrap_or(usize::MAX).min(body.len());
            let slice = body[start..].to_vec();
            Ok(SourceStream {
                total_bytes: Some(slice.len() as u64),
                start_offset: start as u64,
                reader: Box::new(std::io::Cursor::new(slice)),
            })
        }
    }

    /// A three-member model whose bodies are distinct, so a member landing at
    /// the wrong path shows up as a hash failure rather than passing silently.
    fn multi_harness(
        free: u64,
    ) -> (tempfile::TempDir, ModelStore, Downloader, ModelDescriptor, Arc<MultiSource>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store =
            ModelStore::with_probe(dir.path(), Arc::new(FixedDiskSpace(free))).expect("store");

        let members = [
            ("encoder.onnx", vec![1_u8; READ_CHUNK + 5]),
            ("decoder.onnx", vec![2_u8; 2048]),
            ("tokens.txt", b"blank\na\nb\n".to_vec()),
        ];
        let mut files = Vec::new();
        let mut bodies = Vec::new();
        for (name, body) in members {
            let url = format!("https://huggingface.co/test/multi/{name}");
            files.push(ModelFile {
                file_name: name.to_string(),
                url: url.clone(),
                size_bytes: body.len() as u64,
                sha256: sha256_bytes(&body),
            });
            bodies.push((url, body));
        }

        let total: u64 = files.iter().map(|f| f.size_bytes).sum();
        let model = ModelDescriptor {
            id: "multi-model".into(),
            kind: ModelKind::Stt,
            family: "parakeet".into(),
            runtime: ModelRuntime::SherpaOnnx,
            display_name: "Multi".into(),
            quality_tier: 3,
            size_bytes: total,
            runtime_ram_mb: 10,
            languages: vec!["en".into()],
            min_tier: DeviceTier::Low,
            license: "MIT".into(),
            attribution: "test".into(),
            file_name: "multi-model".into(),
            context_tokens: None,
            sha256: "0".repeat(64),
            url: "https://huggingface.co/test/multi".into(),
            files,
        };

        let source = MultiSource::new(bodies);
        let downloader = Downloader::new(store.clone(), source.clone());
        (dir, store, downloader, model, source)
    }

    #[test]
    fn a_multi_file_model_installs_every_member_into_one_directory() {
        let (_d, store, dl, model, _src) = multi_harness(u64::MAX);
        let mut seen = Vec::new();
        let dir = dl
            .download(&model, &CancelToken::new(), &mut |p| seen.push(p.clone()))
            .expect("download");

        assert!(dir.is_dir(), "multi-file model installs as a directory");
        for f in &model.files {
            let path = dir.join(&f.file_name);
            assert!(path.is_file(), "{} missing", f.file_name);
            assert_eq!(sha256_file(&path).expect("hash"), f.sha256);
        }
        store.verify(&model).expect("all members verify");
        assert!(store.is_installed(&model));
        assert_eq!(store.installed_size(&model), Some(model.size_bytes));

        // Progress is about the model, not the current member: it must never
        // go backwards as the downloader moves from one file to the next.
        let mut high = 0;
        for p in &seen {
            assert!(
                p.downloaded_bytes >= high,
                "progress went backwards: {} after {high}",
                p.downloaded_bytes
            );
            high = p.downloaded_bytes;
            assert_eq!(p.total_bytes, model.size_bytes);
        }
        assert!(matches!(
            seen.last().map(|p| &p.state),
            Some(DownloadState::Completed)
        ));
    }

    #[test]
    fn a_second_download_reuses_members_already_on_disk() {
        let (_d, _store, dl, model, src) = multi_harness(u64::MAX);
        dl.download(&model, &CancelToken::new(), &mut |_| {})
            .expect("first");
        let opens = src.opens.load(Ordering::SeqCst);

        dl.download(&model, &CancelToken::new(), &mut |_| {})
            .expect("second");
        assert_eq!(
            src.opens.load(Ordering::SeqCst),
            opens,
            "an installed multi-file model must not be re-fetched"
        );
    }

    #[test]
    fn a_corrupt_member_is_refetched_without_redownloading_its_siblings() {
        let (_d, store, dl, model, src) = multi_harness(u64::MAX);
        dl.download(&model, &CancelToken::new(), &mut |_| {})
            .expect("first");
        let opens_after_first = src.opens.load(Ordering::SeqCst);

        // Corrupt the smallest member only.
        let victim = &model.files[2];
        fs::write(store.member_path(&model, victim), b"tampered").expect("tamper");
        assert!(store.verify(&model).is_err(), "tampering must be detected");

        dl.download(&model, &CancelToken::new(), &mut |_| {})
            .expect("repair");
        store.verify(&model).expect("repaired");

        // Exactly one member was refetched, not all three.
        assert_eq!(
            src.opens.load(Ordering::SeqCst) - opens_after_first,
            1,
            "only the corrupt member should be refetched"
        );
    }

    #[test]
    fn a_multi_file_model_with_one_placeholder_member_installs_nothing() {
        let (_d, store, dl, mut model, src) = multi_harness(u64::MAX);
        model.files[1].sha256 = "0".repeat(64);
        let result = dl.download(&model, &CancelToken::new(), &mut |_| {});
        assert!(matches!(result, Err(AiError::Config(_))));
        assert_eq!(src.opens.load(Ordering::SeqCst), 0, "no bytes should move");
        assert!(!store.is_installed(&model));
    }

    #[test]
    fn deleting_a_multi_file_model_removes_the_whole_directory() {
        let (_d, store, dl, model, _src) = multi_harness(u64::MAX);
        dl.download(&model, &CancelToken::new(), &mut |_| {})
            .expect("download");
        let freed = store.delete(&model).expect("delete");
        assert_eq!(freed, model.size_bytes);
        assert!(!store.is_installed(&model));
        assert!(!store.path_for(&model).exists());
    }
    use crate::models::store::{sha256_bytes, FixedDiskSpace};

    fn body(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 253) as u8).collect()
    }

    fn descriptor(body: &[u8]) -> ModelDescriptor {
        ModelDescriptor {
            id: "test-model".into(),
            kind: ModelKind::Stt,
            family: "test".into(),
            runtime: ModelRuntime::WhisperCpp,
            display_name: "Test".into(),
            quality_tier: 1,
            size_bytes: body.len() as u64,
            runtime_ram_mb: 10,
            languages: vec!["en".into()],
            min_tier: DeviceTier::Low,
            license: "MIT".into(),
            attribution: "test".into(),
            file_name: "test-model.bin".into(),
            context_tokens: None,
            sha256: sha256_bytes(body),
            url: "https://huggingface.co/test/test-model.bin".into(),
            files: Vec::new(),
        }
    }

    struct Harness {
        _dir: tempfile::TempDir,
        store: ModelStore,
        source: Arc<FakeSource>,
        downloader: Downloader,
        model: ModelDescriptor,
        body: Vec<u8>,
    }

    fn harness(size: usize, free: u64) -> Harness {
        let dir = tempfile::tempdir().expect("tempdir");
        let store =
            ModelStore::with_probe(dir.path(), Arc::new(FixedDiskSpace(free))).expect("store");
        let payload = body(size);
        let source = FakeSource::new(payload.clone());
        let downloader = Downloader::new(store.clone(), source.clone());
        let model = descriptor(&payload);
        Harness {
            _dir: dir,
            store,
            source,
            downloader,
            model,
            body: payload,
        }
    }

    fn collect(h: &Harness, cancel: &CancelToken) -> (Result<std::path::PathBuf>, Vec<DownloadProgress>) {
        let mut seen = Vec::new();
        let r = h
            .downloader
            .download(&h.model, cancel, &mut |p| seen.push(p.clone()));
        (r, seen)
    }

    #[test]
    fn a_clean_download_installs_atomically_and_reports_progress() {
        let h = harness(READ_CHUNK * 3 + 11, u64::MAX);
        let (result, progress) = collect(&h, &CancelToken::new());

        let path = result.expect("download");
        assert_eq!(fs::read(&path).expect("read"), h.body);
        assert!(!h.store.part_path_for(&h.model).exists(), ".part must be gone");
        h.store.verify(&h.model).expect("installed file verifies");

        assert!(matches!(
            progress.last().map(|p| &p.state),
            Some(DownloadState::Completed)
        ));
        assert!(progress.iter().any(|p| p.state == DownloadState::Verifying));
        assert!(progress.iter().any(|p| p.state == DownloadState::Running));
        let last = progress.last().expect("progress");
        assert!((last.fraction() - 1.0).abs() < f32::EPSILON);
        assert!(last.is_terminal());
    }

    #[test]
    fn cancellation_stops_the_download_and_keeps_the_partial() {
        let h = harness(READ_CHUNK * 4, u64::MAX);
        let cancel = CancelToken::new();

        let mut seen = Vec::new();
        let result = h.downloader.download(&h.model, &cancel, &mut |p| {
            if p.downloaded_bytes > 0 {
                cancel.cancel();
            }
            seen.push(p.clone());
        });

        assert!(matches!(result, Err(AiError::DownloadCancelled)));
        assert!(matches!(
            seen.last().map(|p| &p.state),
            Some(DownloadState::Cancelled)
        ));
        let partial = h.store.partial_size(&h.model);
        assert!(partial > 0 && partial < h.body.len() as u64, "{partial}");
        assert!(!h.store.is_installed(&h.model));
    }

    #[test]
    fn a_cancelled_download_resumes_from_where_it_stopped() {
        let h = harness(READ_CHUNK * 4, u64::MAX);

        let cancel = CancelToken::new();
        let _ = h.downloader.download(&h.model, &cancel, &mut |p| {
            if p.downloaded_bytes >= READ_CHUNK as u64 {
                cancel.cancel();
            }
        });
        let resumed_from = h.store.partial_size(&h.model);
        assert!(resumed_from > 0);

        let (result, progress) = collect(&h, &CancelToken::new());
        result.expect("resumed download");

        // The transport was asked to start at the resume point…
        assert_eq!(
            h.source.last_offset.load(std::sync::atomic::Ordering::SeqCst) as u64,
            resumed_from
        );
        // …and the first progress report already credits those bytes.
        assert_eq!(
            progress.first().map(|p| p.downloaded_bytes),
            Some(resumed_from)
        );
        assert_eq!(
            fs::read(h.store.path_for(&h.model)).expect("read"),
            h.body,
            "resumed file must be byte-identical"
        );
    }

    #[test]
    fn a_server_that_ignores_range_restarts_cleanly_instead_of_splicing() {
        let h = harness(READ_CHUNK * 3, u64::MAX);

        let cancel = CancelToken::new();
        let _ = h.downloader.download(&h.model, &cancel, &mut |p| {
            if p.downloaded_bytes >= READ_CHUNK as u64 {
                cancel.cancel();
            }
        });
        assert!(h.store.partial_size(&h.model) > 0);

        h.source.set(Behaviour::IgnoresRange);
        let (result, _) = collect(&h, &CancelToken::new());
        result.expect("restart");
        assert_eq!(fs::read(h.store.path_for(&h.model)).expect("read"), h.body);
    }

    #[test]
    fn corrupt_bytes_fail_the_hash_and_the_partial_is_destroyed() {
        let h = harness(4096, u64::MAX);
        h.source.set(Behaviour::ServesCorruptBytes);

        let (result, _) = collect(&h, &CancelToken::new());
        match result {
            Err(AiError::IntegrityMismatch { id, expected, actual }) => {
                assert_eq!(id, "test-model");
                assert_ne!(expected, actual);
            }
            other => panic!("expected IntegrityMismatch, got {other:?}"),
        }
        assert!(!h.store.is_installed(&h.model), "must not install a bad file");
        assert!(
            !h.store.part_path_for(&h.model).exists(),
            "a hash failure must not leave a resumable partial"
        );
    }

    #[test]
    fn a_short_body_is_rejected_before_hashing() {
        let h = harness(4096, u64::MAX);
        h.source.set(Behaviour::TruncatesAfter(100));
        let (result, _) = collect(&h, &CancelToken::new());
        assert!(matches!(result, Err(AiError::Download(_))));
        assert!(!h.store.is_installed(&h.model));
    }

    #[test]
    fn a_transport_failure_surfaces_as_a_retryable_error() {
        let h = harness(4096, u64::MAX);
        h.source.set(Behaviour::Unreachable);
        let (result, _) = collect(&h, &CancelToken::new());
        match result {
            Err(e) => assert!(e.is_retryable(), "{e:?}"),
            Ok(_) => panic!("expected failure"),
        }
    }

    #[test]
    fn the_free_space_preflight_runs_before_any_bytes_move() {
        let h = harness(1_000_000, 1_000);
        let (result, _) = collect(&h, &CancelToken::new());
        assert!(matches!(result, Err(AiError::InsufficientSpace { .. })));
        assert_eq!(h.source.opens.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn a_placeholder_checksum_refuses_to_install_anything() {
        let mut h = harness(1024, u64::MAX);
        h.model.sha256 = "0".repeat(64);
        let (result, _) = collect(&h, &CancelToken::new());
        assert!(matches!(result, Err(AiError::Config(_))));
        assert_eq!(h.source.opens.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn an_off_allowlist_url_never_reaches_the_transport() {
        let mut h = harness(1024, u64::MAX);
        h.model.url = "https://cdn.evil.example/model.bin".into();
        let (result, _) = collect(&h, &CancelToken::new());
        assert!(matches!(result, Err(AiError::HostNotAllowed(_))));
        assert_eq!(h.source.opens.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn an_already_installed_model_is_a_no_op() {
        let h = harness(2048, u64::MAX);
        collect(&h, &CancelToken::new()).0.expect("first download");
        let opens = h.source.opens.load(std::sync::atomic::Ordering::SeqCst);

        let (result, progress) = collect(&h, &CancelToken::new());
        result.expect("second call");
        assert_eq!(
            h.source.opens.load(std::sync::atomic::Ordering::SeqCst),
            opens,
            "must not re-download"
        );
        assert_eq!(progress.len(), 1);
        assert_eq!(progress[0].state, DownloadState::Completed);
    }

    #[test]
    fn a_partial_with_the_wrong_bytes_fails_the_hash_and_recovers_next_time() {
        let h = harness(READ_CHUNK * 2, u64::MAX);
        // A resumable `.part` whose bytes do not belong to this model — e.g. a
        // partially restored backup. Resume splices it and the hash catches it.
        fs::write(h.store.part_path_for(&h.model), vec![0_u8; 32]).expect("write");

        let (result, _) = collect(&h, &CancelToken::new());
        assert!(matches!(result, Err(AiError::IntegrityMismatch { .. })));
        assert!(
            !h.store.part_path_for(&h.model).exists(),
            "the poisoned partial must be destroyed, not left to fail forever"
        );

        // With the bad partial gone, a plain retry succeeds.
        let (result, _) = collect(&h, &CancelToken::new());
        result.expect("retry after the bad partial was destroyed");
        assert_eq!(fs::read(h.store.path_for(&h.model)).expect("read"), h.body);
    }

    #[test]
    fn a_truncated_partial_file_is_rehashed_and_not_trusted_blindly() {
        // `.part` shorter than what `partial_size` reports cannot happen via
        // the normal path, but a truncated restore can produce a `.part` whose
        // bytes are a genuine prefix. That case must resume and succeed.
        let h = harness(READ_CHUNK * 2, u64::MAX);
        let prefix = h.body.get(..1000).unwrap_or(&[]).to_vec();
        fs::write(h.store.part_path_for(&h.model), &prefix).expect("write");

        let (result, progress) = collect(&h, &CancelToken::new());
        result.expect("resume from a genuine prefix");
        assert_eq!(progress.first().map(|p| p.downloaded_bytes), Some(1000));
        assert_eq!(fs::read(h.store.path_for(&h.model)).expect("read"), h.body);
    }

    #[test]
    fn progress_fraction_never_divides_by_zero() {
        let p = DownloadProgress {
            model_id: "x".into(),
            downloaded_bytes: 0,
            total_bytes: 0,
            bytes_per_sec: 0.0,
            eta_secs: None,
            state: DownloadState::Queued,
        };
        assert!((p.fraction() - 0.0).abs() < f32::EPSILON);
        assert!(!p.is_terminal());
    }
}

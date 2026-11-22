//! On-disk model storage.
//!
//! The host supplies the root directory — iOS `Application Support/ai`,
//! Android `context.filesDir/ai`, desktop `app_data_dir()/ai` (design spec
//! §5.1). Nothing here hardcodes a platform path, and the free-space probe is
//! a trait so tests never touch a real filesystem quota.
//!
//! Layout under the root:
//!
//! ```text
//! <root>/models/<file_name>        installed, verified
//! <root>/tmp/<id>.part            in-flight download
//! ```

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{AiError, Result};
use crate::models::catalog::{Catalog, ModelDescriptor, ModelFile};
use crate::util::to_hex;

/// How many bytes to read at a time when hashing.
const HASH_CHUNK: usize = 256 * 1024;

/// Free-space lookup, abstracted so tests can force the preflight to fail.
pub trait DiskSpaceProbe: std::fmt::Debug + Send + Sync {
    /// Bytes available to an unprivileged user on the filesystem holding `path`.
    ///
    /// # Errors
    /// [`AiError::Io`] if the filesystem cannot be interrogated.
    fn available_bytes(&self, path: &Path) -> Result<u64>;
}

/// Always reports a fixed number. Used by tests, and by hosts built without
/// the `disk-space` feature that would rather do the check themselves.
#[derive(Debug, Clone, Copy)]
pub struct FixedDiskSpace(pub u64);

impl DiskSpaceProbe for FixedDiskSpace {
    fn available_bytes(&self, _path: &Path) -> Result<u64> {
        Ok(self.0)
    }
}

/// Real `statvfs`/`GetDiskFreeSpaceEx` probe.
#[cfg(feature = "disk-space")]
#[derive(Debug, Clone, Copy, Default)]
pub struct SysDiskSpace;

#[cfg(feature = "disk-space")]
impl DiskSpaceProbe for SysDiskSpace {
    fn available_bytes(&self, path: &Path) -> Result<u64> {
        // Walk up to the nearest directory that exists — the target may not
        // have been created yet.
        let mut probe = path;
        loop {
            if probe.exists() {
                return fs4::available_space(probe).map_err(|e| AiError::io(probe, &e));
            }
            match probe.parent() {
                Some(p) => probe = p,
                None => {
                    return Err(AiError::io_msg(path, "no existing ancestor directory"));
                }
            }
        }
    }
}

/// A model present on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct InstalledModel {
    pub id: String,
    /// Absolute path, for the inference backend to open.
    pub path: String,
    pub size_bytes: u64,
    /// True if the size matches the catalogue. A full hash is only done on
    /// demand — see [`ModelStore::verify`].
    pub size_matches: bool,
}

/// Model directory management: paths, verification, deletion, preflight.
#[derive(Debug, Clone)]
pub struct ModelStore {
    root: PathBuf,
    probe: Arc<dyn DiskSpaceProbe>,
}

impl ModelStore {
    /// Open a store rooted at a host-supplied directory, using the platform
    /// free-space probe when the `disk-space` feature is on and an
    /// "always plenty" stub otherwise.
    ///
    /// # Errors
    /// [`AiError::Config`] for an empty root; [`AiError::Io`] if the
    /// directories cannot be created.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        #[cfg(feature = "disk-space")]
        let probe: Arc<dyn DiskSpaceProbe> = Arc::new(SysDiskSpace);
        #[cfg(not(feature = "disk-space"))]
        let probe: Arc<dyn DiskSpaceProbe> = Arc::new(FixedDiskSpace(u64::MAX));
        Self::with_probe(root, probe)
    }

    /// Open a store with an explicit free-space probe.
    ///
    /// # Errors
    /// As [`ModelStore::new`].
    pub fn with_probe(root: impl Into<PathBuf>, probe: Arc<dyn DiskSpaceProbe>) -> Result<Self> {
        let root = root.into();
        if root.as_os_str().is_empty() {
            return Err(AiError::Config("model store root is empty".into()));
        }
        let store = Self { root, probe };
        store.ensure_dirs()?;
        Ok(store)
    }

    /// The host-supplied root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where installed models live.
    #[must_use]
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// Where partial downloads live.
    #[must_use]
    pub fn tmp_dir(&self) -> PathBuf {
        self.root.join("tmp")
    }

    /// Create the directory tree if it is not there.
    ///
    /// # Errors
    /// [`AiError::Io`] on failure.
    pub fn ensure_dirs(&self) -> Result<()> {
        for d in [self.models_dir(), self.tmp_dir()] {
            fs::create_dir_all(&d).map_err(|e| AiError::io(&d, &e))?;
        }
        Ok(())
    }

    /// Final resting path of a model: the file for a single-blob model, the
    /// containing directory for a multi-file one. Either way this is what the
    /// inference backend is handed — sherpa-onnx wants the directory.
    #[must_use]
    pub fn path_for(&self, model: &ModelDescriptor) -> PathBuf {
        self.models_dir().join(sanitize_file_name(&model.file_name))
    }

    /// Where one member of a multi-file model lives.
    #[must_use]
    pub fn member_path(&self, model: &ModelDescriptor, file: &ModelFile) -> PathBuf {
        self.path_for(model).join(sanitize_file_name(&file.file_name))
    }

    /// Where one member's in-progress download lives.
    #[must_use]
    pub fn member_part_path(&self, model: &ModelDescriptor, file: &ModelFile) -> PathBuf {
        self.tmp_dir().join(format!(
            "{}.{}.part",
            sanitize_file_name(&model.id),
            sanitize_file_name(&file.file_name)
        ))
    }

    /// Bytes already downloaded for one member.
    #[must_use]
    pub fn member_partial_size(&self, model: &ModelDescriptor, file: &ModelFile) -> u64 {
        fs::metadata(self.member_part_path(model, file))
            .ok()
            .map_or(0, |m| m.len())
    }

    /// Path of the in-progress download.
    #[must_use]
    pub fn part_path_for(&self, model: &ModelDescriptor) -> PathBuf {
        self.tmp_dir()
            .join(format!("{}.part", sanitize_file_name(&model.id)))
    }

    /// Is the model installed? A multi-file model counts only when every
    /// member is present — a half-populated directory is not installed.
    #[must_use]
    pub fn is_installed(&self, model: &ModelDescriptor) -> bool {
        if model.is_multi_file() {
            model
                .files
                .iter()
                .all(|f| self.member_path(model, f).is_file())
        } else {
            self.path_for(model).is_file()
        }
    }

    /// Size on disk, if present. Sums the members for a multi-file model.
    #[must_use]
    pub fn installed_size(&self, model: &ModelDescriptor) -> Option<u64> {
        if model.is_multi_file() {
            if !self.is_installed(model) {
                return None;
            }
            return Some(
                model
                    .files
                    .iter()
                    .filter_map(|f| fs::metadata(self.member_path(model, f)).ok())
                    .map(|m| m.len())
                    .sum(),
            );
        }
        fs::metadata(self.path_for(model)).ok().map(|m| m.len())
    }

    /// Bytes already downloaded into the `.part` file(s). Zero if there are none.
    #[must_use]
    pub fn partial_size(&self, model: &ModelDescriptor) -> u64 {
        if model.is_multi_file() {
            return model
                .files
                .iter()
                .map(|f| self.member_partial_size(model, f))
                .sum();
        }
        fs::metadata(self.part_path_for(model))
            .ok()
            .map_or(0, |m| m.len())
    }

    /// Describe a model on disk.
    ///
    /// # Errors
    /// [`AiError::ModelNotInstalled`] if it is not there.
    pub fn installed_model(&self, model: &ModelDescriptor) -> Result<InstalledModel> {
        let path = self.path_for(model);
        let size = self
            .installed_size(model)
            .ok_or_else(|| AiError::ModelNotInstalled(model.id.clone()))?;
        Ok(InstalledModel {
            id: model.id.clone(),
            path: path.display().to_string(),
            size_bytes: size,
            size_matches: size == model.size_bytes,
        })
    }

    /// Every catalogue model currently on disk.
    #[must_use]
    pub fn installed(&self, catalog: &Catalog) -> Vec<InstalledModel> {
        catalog
            .models
            .iter()
            .filter_map(|m| self.installed_model(m).ok())
            .collect()
    }

    /// Total bytes occupied by installed models plus partials.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        dir_size(&self.models_dir()).saturating_add(dir_size(&self.tmp_dir()))
    }

    /// Refuse to start a download that cannot finish (design spec §6.3).
    ///
    /// # Errors
    /// [`AiError::InsufficientSpace`] when free space is below `size * 1.2`
    /// minus whatever has already been downloaded.
    pub fn preflight(&self, model: &ModelDescriptor) -> Result<()> {
        // Credit both partials and, for a multi-file model, members already
        // installed — otherwise resuming the last 10 MB of a directory would
        // demand room for the whole 660 MB again.
        let mut have = self.partial_size(model);
        if model.is_multi_file() {
            have = have.saturating_add(
                model
                    .files
                    .iter()
                    .filter_map(|f| fs::metadata(self.member_path(model, f)).ok())
                    .map(|m| m.len())
                    .sum(),
            );
        }
        let needed = model.required_free_bytes().saturating_sub(have);
        let available = self.probe.available_bytes(&self.root)?;
        if available < needed {
            return Err(AiError::InsufficientSpace {
                needed_bytes: needed,
                available_bytes: available,
            });
        }
        Ok(())
    }

    /// Stream the installed file through SHA-256 and compare to the catalogue.
    ///
    /// Refuses outright while the catalogue still carries a placeholder
    /// digest — failing closed is the whole point of this check.
    ///
    /// # Errors
    /// [`AiError::ModelNotInstalled`], [`AiError::Config`] for a placeholder
    /// digest, [`AiError::IntegrityMismatch`] on a bad file, [`AiError::Io`].
    pub fn verify(&self, model: &ModelDescriptor) -> Result<()> {
        if !model.has_real_checksum() {
            return Err(AiError::Config(format!(
                "model {} has a placeholder checksum; fill in catalog.json before shipping",
                model.id
            )));
        }
        if model.is_multi_file() {
            for f in &model.files {
                let path = self.member_path(model, f);
                if !path.is_file() {
                    return Err(AiError::ModelNotInstalled(model.id.clone()));
                }
                let actual = sha256_file(&path)?;
                if !actual.eq_ignore_ascii_case(&f.sha256) {
                    return Err(AiError::IntegrityMismatch {
                        id: format!("{}/{}", model.id, f.file_name),
                        expected: f.sha256.to_ascii_lowercase(),
                        actual,
                    });
                }
            }
            return Ok(());
        }

        let path = self.path_for(model);
        if !path.is_file() {
            return Err(AiError::ModelNotInstalled(model.id.clone()));
        }
        let actual = sha256_file(&path)?;
        if actual.eq_ignore_ascii_case(&model.sha256) {
            Ok(())
        } else {
            Err(AiError::IntegrityMismatch {
                id: model.id.clone(),
                expected: model.sha256.to_ascii_lowercase(),
                actual,
            })
        }
    }

    /// Delete an installed model. Returns bytes reclaimed; deleting something
    /// that is not there is not an error.
    ///
    /// # Errors
    /// [`AiError::Io`] if the file exists but cannot be removed.
    pub fn delete(&self, model: &ModelDescriptor) -> Result<u64> {
        let path = self.path_for(model);
        if model.is_multi_file() {
            let size = dir_size(&path);
            return match fs::remove_dir_all(&path) {
                Ok(()) => Ok(size),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
                Err(e) => Err(AiError::io(&path, &e)),
            };
        }
        let size = fs::metadata(&path).map_or(0, |m| m.len());
        match fs::remove_file(&path) {
            Ok(()) => Ok(size),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(e) => Err(AiError::io(&path, &e)),
        }
    }

    /// Throw away a partial download.
    ///
    /// # Errors
    /// [`AiError::Io`] if the file exists but cannot be removed.
    pub fn discard_partial(&self, model: &ModelDescriptor) -> Result<()> {
        let path = self.part_path_for(model);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(AiError::io(&path, &e)),
        }
    }

    /// Delete `.part` files older than `max_age` (design spec §6.4).
    /// Returns how many were removed.
    ///
    /// # Errors
    /// [`AiError::Io`] if the tmp directory cannot be listed.
    pub fn cleanup_stale_parts(&self, max_age: Duration) -> Result<usize> {
        let dir = self.tmp_dir();
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(AiError::io(&dir, &e)),
        };
        let now = SystemTime::now();
        let mut removed = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "part") {
                continue;
            }
            let age = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| now.duration_since(t).ok())
                .unwrap_or_default();
            if age >= max_age && fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Empty the tmp directory. Called on every launch (design spec §5.2) so
    /// no spill audio or partial download outlives the process that made it.
    ///
    /// # Errors
    /// [`AiError::Io`] if the directory cannot be recreated.
    pub fn clear_tmp(&self) -> Result<()> {
        let dir = self.tmp_dir();
        if dir.exists() {
            fs::remove_dir_all(&dir).map_err(|e| AiError::io(&dir, &e))?;
        }
        fs::create_dir_all(&dir).map_err(|e| AiError::io(&dir, &e))
    }
}

/// SHA-256 a file, streaming so a 2.5 GB GGUF never lands in RAM.
///
/// # Errors
/// [`AiError::Io`] if the file cannot be read.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).map_err(|e| AiError::io(path, &e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; HASH_CHUNK];
    loop {
        let n = file.read(&mut buf).map_err(|e| AiError::io(path, &e))?;
        if n == 0 {
            break;
        }
        hasher.update(buf.get(..n).unwrap_or(&[]));
    }
    Ok(to_hex(&hasher.finalize()))
}

/// SHA-256 of a byte slice, as lowercase hex.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    to_hex(&hasher.finalize())
}

/// Strip anything that could escape the store directory. Catalogue values are
/// trusted (they are compiled in) but this is cheap insurance if a signed
/// remote catalogue ever lands.
fn sanitize_file_name(name: &str) -> String {
    let mut cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    // Collapse every `..` so no traversal token survives even as text.
    while cleaned.contains("..") {
        cleaned = cleaned.replace("..", "_");
    }
    let trimmed = cleaned.trim_start_matches('.');
    if trimmed.is_empty() {
        "model.bin".to_string()
    } else {
        trimmed.to_string()
    }
}

fn dir_size(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(std::fs::Metadata::is_file)
        .map(|m| m.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::catalog::{ModelKind, ModelRuntime};
    use crate::models::capability::DeviceTier;

    fn descriptor(id: &str, size: u64, sha: &str) -> ModelDescriptor {
        ModelDescriptor {
            id: id.to_string(),
            kind: ModelKind::Stt,
            family: "test".into(),
            runtime: ModelRuntime::WhisperCpp,
            display_name: "Test".into(),
            quality_tier: 1,
            size_bytes: size,
            runtime_ram_mb: 10,
            languages: vec!["en".into()],
            min_tier: DeviceTier::Low,
            license: "MIT".into(),
            attribution: "test".into(),
            file_name: format!("{id}.bin"),
            context_tokens: None,
            sha256: sha.to_string(),
            url: format!("https://huggingface.co/test/{id}.bin"),
            files: Vec::new(),
        }
    }

    fn store() -> (tempfile::TempDir, ModelStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = ModelStore::with_probe(dir.path(), Arc::new(FixedDiskSpace(u64::MAX)))
            .expect("store");
        (dir, s)
    }

    #[test]
    fn directories_are_created_under_the_host_supplied_root() {
        let (dir, s) = store();
        assert!(s.models_dir().is_dir());
        assert!(s.tmp_dir().is_dir());
        assert!(s.models_dir().starts_with(dir.path()));
    }

    #[test]
    fn an_empty_root_is_rejected() {
        assert!(matches!(ModelStore::new(""), Err(AiError::Config(_))));
    }

    #[test]
    fn paths_are_derived_from_the_catalogue_not_guessed() {
        let (_d, s) = store();
        let m = descriptor("m1", 10, "aa");
        assert_eq!(s.path_for(&m).file_name().and_then(|n| n.to_str()), Some("m1.bin"));
        assert_eq!(
            s.part_path_for(&m).file_name().and_then(|n| n.to_str()),
            Some("m1.part")
        );
    }

    #[test]
    fn traversal_in_a_file_name_cannot_escape_the_store() {
        let (_d, s) = store();
        let mut m = descriptor("evil", 10, "aa");
        m.file_name = "../../../../etc/passwd".into();
        let p = s.path_for(&m);
        assert!(p.starts_with(s.models_dir()), "{}", p.display());
        assert!(!p.display().to_string().contains(".."));
    }

    #[test]
    fn install_detection_and_deletion_report_reclaimed_space() {
        let (_d, s) = store();
        let m = descriptor("m1", 5, "aa");
        assert!(!s.is_installed(&m));
        assert_eq!(s.delete(&m).expect("delete missing"), 0);

        fs::write(s.path_for(&m), b"hello").expect("write");
        assert!(s.is_installed(&m));
        assert_eq!(s.installed_size(&m), Some(5));
        assert_eq!(s.delete(&m).expect("delete"), 5);
        assert!(!s.is_installed(&m));
    }

    #[test]
    fn verify_accepts_a_good_file_and_rejects_a_corrupt_one() {
        let (_d, s) = store();
        let body = b"the quick brown fox";
        let sha = sha256_bytes(body);
        let m = descriptor("m1", body.len() as u64, &sha);

        fs::write(s.path_for(&m), body).expect("write");
        s.verify(&m).expect("verify good file");

        fs::write(s.path_for(&m), b"tampered").expect("write");
        match s.verify(&m) {
            Err(AiError::IntegrityMismatch { expected, .. }) => assert_eq!(expected, sha),
            other => panic!("expected mismatch, got {other:?}"),
        }
    }

    #[test]
    fn verify_fails_closed_on_a_placeholder_checksum() {
        let (_d, s) = store();
        let m = descriptor("m1", 5, &"0".repeat(64));
        fs::write(s.path_for(&m), b"hello").expect("write");
        assert!(matches!(s.verify(&m), Err(AiError::Config(_))));
    }

    #[test]
    fn verify_of_a_missing_file_is_an_error_not_a_panic() {
        let (_d, s) = store();
        let m = descriptor("m1", 5, &sha256_bytes(b"x"));
        assert!(matches!(s.verify(&m), Err(AiError::ModelNotInstalled(_))));
    }

    #[test]
    fn preflight_requires_size_plus_twenty_percent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let m = descriptor("m1", 1_000_000, "aa");

        let tight = ModelStore::with_probe(dir.path(), Arc::new(FixedDiskSpace(1_100_000)))
            .expect("store");
        match tight.preflight(&m) {
            Err(AiError::InsufficientSpace { needed_bytes, .. }) => {
                assert_eq!(needed_bytes, 1_200_000);
            }
            other => panic!("expected InsufficientSpace, got {other:?}"),
        }

        let roomy = ModelStore::with_probe(dir.path(), Arc::new(FixedDiskSpace(1_200_000)))
            .expect("store");
        roomy.preflight(&m).expect("enough room");
    }

    #[test]
    fn preflight_credits_bytes_already_downloaded() {
        let dir = tempfile::tempdir().expect("tempdir");
        let s = ModelStore::with_probe(dir.path(), Arc::new(FixedDiskSpace(300_000)))
            .expect("store");
        let m = descriptor("m1", 1_000_000, "aa");
        assert!(s.preflight(&m).is_err());

        fs::write(s.part_path_for(&m), vec![0_u8; 950_000]).expect("write part");
        s.preflight(&m).expect("mostly downloaded already");
    }

    #[test]
    fn stale_part_cleanup_leaves_fresh_partials_alone() {
        let (_d, s) = store();
        let m = descriptor("m1", 10, "aa");
        fs::write(s.part_path_for(&m), b"partial").expect("write");
        fs::write(s.tmp_dir().join("notapart.txt"), b"x").expect("write");

        assert_eq!(s.cleanup_stale_parts(Duration::from_hours(1)).expect("clean"), 0);
        assert!(s.part_path_for(&m).exists());

        assert_eq!(s.cleanup_stale_parts(Duration::ZERO).expect("clean"), 1);
        assert!(!s.part_path_for(&m).exists());
        assert!(s.tmp_dir().join("notapart.txt").exists());
    }

    #[test]
    fn clear_tmp_empties_but_keeps_the_directory() {
        let (_d, s) = store();
        fs::write(s.tmp_dir().join("spill.pcm"), b"audio").expect("write");
        s.clear_tmp().expect("clear");
        assert!(s.tmp_dir().is_dir());
        assert_eq!(fs::read_dir(s.tmp_dir()).into_iter().flatten().count(), 0);
    }

    #[test]
    fn total_bytes_counts_models_and_partials() {
        let (_d, s) = store();
        let m = descriptor("m1", 4, "aa");
        fs::write(s.path_for(&m), b"1234").expect("write");
        fs::write(s.part_path_for(&m), b"567").expect("write");
        assert_eq!(s.total_bytes(), 7);
    }

    #[test]
    fn installed_lists_only_what_is_really_there() {
        let (_d, s) = store();
        let catalog = Catalog::bundled().expect("catalog");
        assert_eq!(s.installed(catalog), []);

        let m = catalog.get("whisper-tiny-q5_1").expect("model");
        fs::write(s.path_for(m), b"weights").expect("write");
        let installed = s.installed(catalog);
        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].id, "whisper-tiny-q5_1");
        assert!(!installed[0].size_matches, "7 bytes is not 32 MB");
    }

    #[test]
    fn sha256_of_a_file_matches_the_in_memory_digest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("blob");
        // Bigger than one hash chunk, to exercise the streaming loop.
        let body: Vec<u8> = (0..(HASH_CHUNK * 2 + 17)).map(|i| (i % 251) as u8).collect();
        fs::write(&path, &body).expect("write");
        assert_eq!(sha256_file(&path).expect("hash"), sha256_bytes(&body));
    }
}

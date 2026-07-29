//! The model catalogue — bundled in the binary, never fetched.
//!
//! Per design spec §6.1, a compromised or unreachable CDN cannot inject a
//! model: the id, the size, the URL and above all the SHA-256 all come from
//! the signed app binary.
//!
//! ## Where the weights come from
//!
//! Every entry points at the publisher's own Hugging Face repository, and
//! every `sha256` is that file's real digest — taken from the HF tree API,
//! where the LFS `oid` *is* the SHA-256. [`ModelDescriptor::has_real_checksum`]
//! still reports an all-zero placeholder and the downloader fails closed on
//! one, so a half-filled catalogue can never install an unverified file.
//!
//! ## Multi-file models
//!
//! sherpa-onnx models are not a single blob: Parakeet is an encoder, a
//! decoder, a joiner and a token list that must sit in one directory. Those
//! entries carry [`ModelDescriptor::files`], each member with its own URL,
//! size and digest, and install into `models/<id>/`. Downloading the members
//! individually beats shipping a `.tar.bz2`: each one is range-resumable and
//! integrity-checked on its own, and nothing needs twice the disk to unpack.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::error::{AiError, Result};
use crate::models::capability::{DeviceCapability, DeviceTier};

/// Hosts the downloader will talk to. Compile-time, not configuration — this
/// is the enforcement point for the privacy guarantee in design spec §5.4.
pub const ALLOWED_HOSTS: &[&str] = &["huggingface.co"];

/// Domain suffixes that are also allowed, matched on whole labels only.
///
/// `huggingface.co` answers a weights request with a 302 to a signed, expiring
/// URL on its content CDN, and the exact host varies by region and storage
/// backend (`us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`, `cdn-lfs-us-1…`).
/// Pinning the individual hostnames would break the moment HF adds a region,
/// so the CDN is allowed by suffix. Because the match requires a leading dot,
/// `evil-hf.co` and `hf.co.attacker.net` do not qualify.
pub const ALLOWED_HOST_SUFFIXES: &[&str] = &[".hf.co", ".huggingface.co"];

const BUNDLED_JSON: &str = include_str!("catalog.json");

static BUNDLED: OnceLock<std::result::Result<Catalog, String>> = OnceLock::new();

/// What a model is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// Speech to text.
    Stt,
    /// Summarization.
    Llm,
    /// Voice activity detection.
    Vad,
}

/// Which inference runtime loads this file. Drives which Cargo feature is
/// required to actually use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[serde(rename_all = "kebab-case")]
pub enum ModelRuntime {
    /// `sherpa-onnx` via `sherpa-rs` — feature `stt-parakeet`.
    SherpaOnnx,
    /// `whisper.cpp` via `whisper-rs` — feature `stt-whisper`.
    WhisperCpp,
    /// `llama.cpp` via `llama-cpp-2` — feature `llm-llama`.
    LlamaCpp,
}

impl ModelRuntime {
    /// The Cargo feature that must be on for this runtime to exist.
    #[must_use]
    pub const fn feature_name(self) -> &'static str {
        match self {
            Self::SherpaOnnx => "stt-parakeet",
            Self::WhisperCpp => "stt-whisper",
            Self::LlamaCpp => "llm-llama",
        }
    }
}

/// One member file of a multi-file model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct ModelFile {
    /// Name on disk, inside the model's directory.
    pub file_name: String,
    pub url: String,
    pub size_bytes: u64,
    /// Lowercase hex SHA-256 of this member.
    pub sha256: String,
}

/// One downloadable model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct ModelDescriptor {
    pub id: String,
    pub kind: ModelKind,
    pub family: String,
    pub runtime: ModelRuntime,
    /// Plain-language name shown in the picker. Never show a bare model id.
    pub display_name: String,
    /// 1 (fastest) … 4 (best).
    pub quality_tier: u8,
    pub size_bytes: u64,
    /// Peak resident memory when loaded.
    pub runtime_ram_mb: u32,
    pub languages: Vec<String>,
    pub min_tier: DeviceTier,
    pub license: String,
    /// Attribution string that must appear in the About screen for CC-BY
    /// weights.
    pub attribution: String,
    /// Name on disk once installed.
    pub file_name: String,
    /// LLM context window we will actually use. `None` for non-LLM models.
    pub context_tokens: Option<u32>,
    /// Lowercase hex SHA-256 of the downloaded file. Ignored, and conventionally
    /// the placeholder, when [`Self::files`] is non-empty.
    pub sha256: String,
    pub url: String,
    /// Member files, for models that are a directory rather than one blob.
    /// Empty for the single-file case, which is most of the catalogue.
    #[serde(default)]
    pub files: Vec<ModelFile>,
}

const PLACEHOLDER_SHA: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Is this a usable, non-placeholder SHA-256?
fn is_real_sha(sha: &str) -> bool {
    sha.len() == 64 && sha != PLACEHOLDER_SHA && sha.chars().all(|c| c.is_ascii_hexdigit())
}

impl ModelFile {
    /// False while this member still carries the all-zero placeholder.
    #[must_use]
    pub fn has_real_checksum(&self) -> bool {
        is_real_sha(&self.sha256)
    }
}

impl ModelDescriptor {
    /// Does this model install as a directory of files rather than one blob?
    #[must_use]
    pub const fn is_multi_file(&self) -> bool {
        !self.files.is_empty()
    }

    /// False while the catalogue still carries the all-zero placeholder. For a
    /// multi-file model every member must have a real digest — one unverifiable
    /// member makes the whole directory untrustworthy.
    #[must_use]
    pub fn has_real_checksum(&self) -> bool {
        if self.is_multi_file() {
            self.files.iter().all(ModelFile::has_real_checksum)
        } else {
            is_real_sha(&self.sha256)
        }
    }

    /// Bytes that must be free before starting a download: the file plus 20%.
    #[must_use]
    pub const fn required_free_bytes(&self) -> u64 {
        self.size_bytes.saturating_add(self.size_bytes / 5)
    }

    /// Can this device load it at all?
    #[must_use]
    pub fn fits_device(&self, cap: &DeviceCapability) -> bool {
        cap.tier() >= self.min_tier && self.runtime_ram_mb <= cap.ram_budget_mb()
    }

    /// Human-readable reason this model is unavailable, if it is. Users
    /// understand a locked door; they do not understand a missing option
    /// (design spec §6.2).
    #[must_use]
    pub fn unavailable_reason(&self, cap: &DeviceCapability) -> Option<String> {
        if cap.tier() < self.min_tier {
            return Some(format!(
                "Needs a {} device — this one is {}",
                self.min_tier.label(),
                cap.tier().label()
            ));
        }
        if self.runtime_ram_mb > cap.ram_budget_mb() {
            return Some(format!(
                "Needs {} MB of memory — only {} MB is safely available",
                self.runtime_ram_mb,
                cap.ram_budget_mb()
            ));
        }
        None
    }
}

/// A catalogue entry paired with whether this device may use it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct ModelAvailability {
    pub model: ModelDescriptor,
    pub available: bool,
    /// Populated when `available` is false.
    pub reason: Option<String>,
}

/// The whole catalogue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct Catalog {
    pub catalog_version: u32,
    pub models: Vec<ModelDescriptor>,
}

impl Catalog {
    /// The catalogue compiled into this binary.
    ///
    /// # Errors
    /// [`AiError::Internal`] only if `catalog.json` itself is malformed,
    /// which is a build-time bug. Returned rather than panicked because this
    /// is reachable across FFI.
    pub fn bundled() -> Result<&'static Self> {
        match BUNDLED.get_or_init(|| Self::from_json(BUNDLED_JSON).map_err(|e| e.to_string())) {
            Ok(c) => Ok(c),
            Err(e) => Err(AiError::Internal(format!("bundled catalog is invalid: {e}"))),
        }
    }

    /// Parse a catalogue from JSON and validate it.
    ///
    /// # Errors
    /// [`AiError::Config`] for malformed JSON, duplicate ids, or a URL whose
    /// host is not on [`ALLOWED_HOSTS`].
    pub fn from_json(json: &str) -> Result<Self> {
        let catalog: Self = serde_json::from_str(json)
            .map_err(|e| AiError::Config(format!("catalog JSON: {e}")))?;

        let mut seen = std::collections::HashSet::new();
        for m in &catalog.models {
            if !seen.insert(m.id.as_str()) {
                return Err(AiError::Config(format!("duplicate model id: {}", m.id)));
            }
            if m.size_bytes == 0 {
                return Err(AiError::Config(format!("model {} has size 0", m.id)));
            }
            if m.is_multi_file() {
                let mut members = std::collections::HashSet::new();
                for f in &m.files {
                    if f.size_bytes == 0 {
                        return Err(AiError::Config(format!(
                            "model {} member {} has size 0",
                            m.id, f.file_name
                        )));
                    }
                    if !members.insert(f.file_name.as_str()) {
                        return Err(AiError::Config(format!(
                            "model {} has duplicate member {}",
                            m.id, f.file_name
                        )));
                    }
                    check_host(&f.url)?;
                }
                // size_bytes drives the free-space preflight, so it has to be
                // the whole directory, not one member.
                let total: u64 = m.files.iter().map(|f| f.size_bytes).sum();
                if m.size_bytes != total {
                    return Err(AiError::Config(format!(
                        "model {} declares {} bytes but its members total {total}",
                        m.id, m.size_bytes
                    )));
                }
            } else {
                check_host(&m.url)?;
            }
        }
        Ok(catalog)
    }

    /// Look a model up by id.
    ///
    /// # Errors
    /// [`AiError::UnknownModel`] if there is no such id.
    pub fn get(&self, id: &str) -> Result<&ModelDescriptor> {
        self.models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| AiError::UnknownModel(id.to_string()))
    }

    /// All models of a kind, best quality last.
    #[must_use]
    pub fn by_kind(&self, kind: ModelKind) -> Vec<&ModelDescriptor> {
        let mut v: Vec<&ModelDescriptor> =
            self.models.iter().filter(|m| m.kind == kind).collect();
        v.sort_by_key(|m| (m.quality_tier, m.id.clone()));
        v
    }

    /// The whole catalogue annotated for a device — nothing is hidden, models
    /// that do not fit carry a reason.
    #[must_use]
    pub fn availability(&self, cap: &DeviceCapability) -> Vec<ModelAvailability> {
        self.models
            .iter()
            .map(|m| {
                let reason = m.unavailable_reason(cap);
                ModelAvailability {
                    model: m.clone(),
                    available: reason.is_none(),
                    reason,
                }
            })
            .collect()
    }

    /// The best model of `kind` this device can actually run, if any.
    #[must_use]
    pub fn recommended(&self, kind: ModelKind, cap: &DeviceCapability) -> Option<&ModelDescriptor> {
        self.by_kind(kind)
            .into_iter()
            .rfind(|m| m.fits_device(cap))
    }
}

/// Reject any URL that is not HTTPS on an allowlisted host.
///
/// # Errors
/// [`AiError::HostNotAllowed`] for a bad scheme, a bad host, or a URL with
/// embedded credentials.
pub fn check_host(url: &str) -> Result<()> {
    let Some(rest) = url.strip_prefix("https://") else {
        return Err(AiError::HostNotAllowed(format!("not https: {url}")));
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.is_empty() || authority.contains('@') {
        return Err(AiError::HostNotAllowed(url.to_string()));
    }
    // Strip an explicit port before comparing.
    let host = authority.split(':').next().unwrap_or("");
    if host_is_allowed(host) {
        Ok(())
    } else {
        Err(AiError::HostNotAllowed(host.to_string()))
    }
}

/// Is `host` on the allowlist, exactly or by permitted suffix?
#[must_use]
pub fn host_is_allowed(host: &str) -> bool {
    if host.is_empty() {
        return false;
    }
    if ALLOWED_HOSTS.iter().any(|h| h.eq_ignore_ascii_case(host)) {
        return true;
    }
    ALLOWED_HOST_SUFFIXES.iter().any(|suffix| {
        // `len >` not `>=`: the suffix carries a leading dot, so this also
        // rejects a host that is nothing but the bare suffix.
        host.len() > suffix.len()
            && host
                .get(host.len() - suffix.len()..)
                .is_some_and(|tail| tail.eq_ignore_ascii_case(suffix))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::capability::Platform;

    fn phone(total_ram_mb: u32) -> DeviceCapability {
        DeviceCapability {
            platform: Platform::Ios,
            total_ram_mb,
            available_ram_mb: total_ram_mb / 2,
            cpu_cores: 6,
            is_64bit: true,
            has_gpu_backend: true,
        }
    }

    #[test]
    fn the_bundled_catalog_parses_and_validates() {
        let c = Catalog::bundled().expect("bundled catalog must parse");
        assert!(c.catalog_version >= 1);
        assert!(!c.models.is_empty());
        assert!(c.get("whisper-base-q5_1").is_ok());
        assert!(matches!(
            c.get("nope"),
            Err(AiError::UnknownModel(_))
        ));
    }

    #[test]
    fn every_bundled_url_is_https_on_an_allowlisted_host() {
        let c = Catalog::bundled().expect("catalog");
        assert!(!c.models.is_empty(), "nothing to check");
        for m in &c.models {
            check_host(&m.url).unwrap_or_else(|e| panic!("{}: {e}", m.id));
        }
    }

    #[test]
    fn every_bundled_model_carries_a_real_checksum() {
        // The downloader fails closed on a placeholder, so a model that still
        // has one is a model nobody can install. Catch that here rather than
        // on a user's device.
        let c = Catalog::bundled().expect("catalog");
        for m in &c.models {
            assert!(
                m.has_real_checksum(),
                "{} still has a placeholder checksum",
                m.id
            );
        }
    }

    #[test]
    fn a_multi_file_model_declares_the_total_of_its_members() {
        let c = Catalog::bundled().expect("catalog");
        for m in c.models.iter().filter(|m| m.is_multi_file()) {
            let total: u64 = m.files.iter().map(|f| f.size_bytes).sum();
            assert_eq!(m.size_bytes, total, "{} size is not its member total", m.id);
            for f in &m.files {
                assert!(f.has_real_checksum(), "{}/{}", m.id, f.file_name);
            }
        }
    }

    #[test]
    fn host_check_rejects_the_obvious_attacks() {
        assert!(check_host("http://huggingface.co/x").is_err());
        assert!(check_host("https://evil.example.com/x").is_err());
        assert!(check_host("https://huggingface.co.evil.com/x").is_err());
        assert!(check_host("https://user@huggingface.co/x").is_err());
        assert!(check_host("https:///x").is_err());
        assert!(check_host("").is_err());
        assert!(check_host("https://HuggingFace.CO/x").is_ok());
        assert!(check_host("https://huggingface.co:443/x").is_ok());
    }

    #[test]
    fn the_cdn_suffix_rule_matches_whole_labels_only() {
        // The weights CDN is allowed by suffix because HF picks the region at
        // redirect time; that must not become a wildcard.
        assert!(check_host("https://us.aws.cdn.hf.co/xet-bridge-us/abc").is_ok());
        assert!(check_host("https://cas-bridge.xethub.hf.co/x").is_ok());
        assert!(check_host("https://cdn-lfs-us-1.huggingface.co/x").is_ok());

        // Same characters, different owner.
        assert!(check_host("https://evil-hf.co/x").is_err());
        assert!(check_host("https://hf.co.attacker.net/x").is_err());
        assert!(check_host("https://nothf.co/x").is_err());
        // The bare suffix itself is not a host we allow.
        assert!(check_host("https://hf.co/x").is_err());
    }

    #[test]
    fn a_catalog_with_a_duplicate_id_is_rejected() {
        let json = r#"{"catalog_version":1,"models":[
          {"id":"a","kind":"stt","family":"f","runtime":"whisper-cpp","display_name":"A",
           "quality_tier":1,"size_bytes":10,"runtime_ram_mb":10,"languages":["en"],
           "min_tier":"low","license":"MIT","attribution":"x","file_name":"a.bin",
           "context_tokens":null,"sha256":"aa","url":"https://huggingface.co/a"},
          {"id":"a","kind":"stt","family":"f","runtime":"whisper-cpp","display_name":"A2",
           "quality_tier":1,"size_bytes":10,"runtime_ram_mb":10,"languages":["en"],
           "min_tier":"low","license":"MIT","attribution":"x","file_name":"a2.bin",
           "context_tokens":null,"sha256":"aa","url":"https://huggingface.co/a2"}
        ]}"#;
        assert!(matches!(Catalog::from_json(json), Err(AiError::Config(_))));
    }

    #[test]
    fn a_catalog_pointing_off_the_allowlist_is_rejected() {
        let json = r#"{"catalog_version":1,"models":[
          {"id":"a","kind":"stt","family":"f","runtime":"whisper-cpp","display_name":"A",
           "quality_tier":1,"size_bytes":10,"runtime_ram_mb":10,"languages":["en"],
           "min_tier":"low","license":"MIT","attribution":"x","file_name":"a.bin",
           "context_tokens":null,"sha256":"aa","url":"https://cdn.evil.com/a"}
        ]}"#;
        assert!(matches!(
            Catalog::from_json(json),
            Err(AiError::HostNotAllowed(_))
        ));
    }

    #[test]
    fn gating_greys_out_rather_than_hides_and_explains_why() {
        let c = Catalog::bundled().expect("catalog");
        let small = phone(4_096); // Low tier
        let rows = c.availability(&small);
        assert_eq!(rows.len(), c.models.len(), "nothing may be hidden");

        let big_llm = rows
            .iter()
            .find(|r| r.model.id == "qwen3-4b-instruct-2507-q4_k_m")
            .expect("row");
        assert!(!big_llm.available);
        assert!(big_llm.reason.as_deref().unwrap_or("").contains("device"));

        let tiny_stt = rows
            .iter()
            .find(|r| r.model.id == "whisper-tiny-q5_1")
            .expect("row");
        assert!(tiny_stt.available, "{:?}", tiny_stt.reason);
    }

    #[test]
    fn a_low_ram_budget_blocks_a_model_even_at_the_right_tier() {
        let mut d = phone(8_192); // High tier
        d.available_ram_mb = 400; // budget = 200 MB
        let c = Catalog::bundled().expect("catalog");
        let base = c.get("whisper-base-q5_1").expect("model");
        assert!(!base.fits_device(&d));
        assert!(base
            .unavailable_reason(&d)
            .unwrap_or_default()
            .contains("memory"));
    }

    #[test]
    fn recommendation_picks_the_best_that_fits() {
        let c = Catalog::bundled().expect("catalog");
        let low = phone(4_096);
        let rec = c.recommended(ModelKind::Stt, &low).expect("some stt");
        assert!(rec.fits_device(&low));
        assert_eq!(rec.min_tier, DeviceTier::Low);

        // Low-tier devices are offered no summarizer at all (spec §7.3).
        assert!(c.recommended(ModelKind::Llm, &low).is_none());

        let desktop = DeviceCapability::desktop_default();
        let best = c.recommended(ModelKind::Llm, &desktop).expect("llm");
        assert_eq!(best.id, "qwen3-4b-instruct-2507-q4_k_m");
    }

    #[test]
    fn free_space_requirement_includes_headroom() {
        let c = Catalog::bundled().expect("catalog");
        let m = c.get("whisper-base-q5_1").expect("model");
        // Derived from the descriptor rather than restated, so the real file
        // size can change without this becoming a lie.
        assert_eq!(
            m.required_free_bytes(),
            m.size_bytes + m.size_bytes / 5
        );
        assert!(m.required_free_bytes() > m.size_bytes);
    }
}

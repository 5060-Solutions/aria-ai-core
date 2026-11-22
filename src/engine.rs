//! `AiEngine` — the one object each platform binds to (design spec §2.4).
//!
//! It owns the catalogue, the device-capability gating, the model store, the
//! download workers and the live call sessions. Progress and job state are
//! **polled**, not pushed, exactly as the spec chose: Android never wired up
//! `setEventHandler`, both mobile apps already poll for call state, and the
//! desktop has a single event bus it would rather not grow.
//!
//! Deferred to later phases: the `SQLite` + AES-GCM `InsightStore` (§5), the
//! serial post-call job queue (§3.6), thermal/battery policy (§7.3) and the
//! consent state machine (§5.5). Those all sit *above* this type.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::error::{AiError, Result};
use crate::llm::Summarizer;
use crate::models::catalog::{Catalog, ModelAvailability, ModelDescriptor, ModelKind};
use crate::models::download::{
    CancelToken, DownloadProgress, DownloadState, Downloader, ModelSource,
};
use crate::models::store::{InstalledModel, ModelStore};
use crate::models::DeviceCapability;
use crate::session::{CallSession, SessionConfig};
use crate::stt::Transcriber;
use crate::util::lock;

/// Host-supplied setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiConfig {
    /// Root directory for models and temporary files. The host owns this —
    /// iOS `Application Support/ai`, Android `filesDir/ai`, desktop
    /// `app_data_dir()/ai`. Nothing in the crate guesses a path.
    pub storage_root: PathBuf,
    /// What this device can run.
    pub capability: DeviceCapability,
    /// Sent on model downloads.
    pub user_agent: String,
}

impl AiConfig {
    /// Config for a storage root and a device.
    #[must_use]
    pub fn new(storage_root: impl Into<PathBuf>, capability: DeviceCapability) -> Self {
        Self {
            storage_root: storage_root.into(),
            capability,
            user_agent: concat!("aria-ai-core/", env!("CARGO_PKG_VERSION")).to_string(),
        }
    }
}

#[derive(Debug)]
struct DownloadJob {
    cancel: CancelToken,
    progress: Arc<Mutex<DownloadProgress>>,
    handle: Option<JoinHandle<()>>,
}

/// The facade.
#[derive(Debug)]
pub struct AiEngine {
    config: AiConfig,
    catalog: &'static Catalog,
    store: ModelStore,
    source: Arc<dyn ModelSource>,
    jobs: Mutex<HashMap<String, DownloadJob>>,
    sessions: Mutex<HashMap<String, Arc<CallSession>>>,
}

impl AiEngine {
    /// Build an engine using the built-in HTTPS transport.
    ///
    /// # Errors
    /// [`AiError::Config`] if the storage root is unusable or the TLS stack
    /// cannot be initialised.
    #[cfg(feature = "net")]
    pub fn new(config: AiConfig) -> Result<Self> {
        let source = Arc::new(crate::models::http::HttpModelSource::new(
            config.user_agent.clone(),
        )?);
        Self::with_source(config, source)
    }

    /// Build an engine over a caller-supplied transport. This is how tests
    /// run without a network, and how a host can supply its own downloader.
    ///
    /// # Errors
    /// [`AiError::Config`] / [`AiError::Io`] if the storage root is unusable.
    pub fn with_source(config: AiConfig, source: Arc<dyn ModelSource>) -> Result<Self> {
        let store = ModelStore::new(config.storage_root.clone())?;
        // Nothing in tmp survives a restart (design spec §5.2).
        store.clear_tmp()?;
        Ok(Self {
            config,
            catalog: Catalog::bundled()?,
            store,
            source,
            jobs: Mutex::new(HashMap::new()),
            sessions: Mutex::new(HashMap::new()),
        })
    }

    // ── capability & catalogue ──────────────────────────────────────────────

    /// What the host told us about this device.
    #[must_use]
    pub const fn capability(&self) -> &DeviceCapability {
        &self.config.capability
    }

    /// The bundled catalogue.
    #[must_use]
    pub const fn catalog(&self) -> &'static Catalog {
        self.catalog
    }

    /// The model store, for hosts that want to report disk usage.
    #[must_use]
    pub const fn store(&self) -> &ModelStore {
        &self.store
    }

    /// Every catalogue entry annotated for this device. Nothing is hidden;
    /// unavailable rows carry a plain-language reason.
    #[must_use]
    pub fn availability(&self) -> Vec<ModelAvailability> {
        self.catalog.availability(&self.config.capability)
    }

    /// The best model of a kind this device can run, if any.
    #[must_use]
    pub fn recommended(&self, kind: ModelKind) -> Option<&'static ModelDescriptor> {
        self.catalog.recommended(kind, &self.config.capability)
    }

    /// Models present on disk.
    #[must_use]
    pub fn installed_models(&self) -> Vec<InstalledModel> {
        self.store.installed(self.catalog)
    }

    /// Bytes used by models and partial downloads.
    #[must_use]
    pub fn storage_bytes(&self) -> u64 {
        self.store.total_bytes()
    }

    /// Look up a catalogue entry.
    ///
    /// # Errors
    /// [`AiError::UnknownModel`] if there is no such id.
    pub fn descriptor(&self, model_id: &str) -> Result<&'static ModelDescriptor> {
        self.catalog.get(model_id)
    }

    // ── model lifecycle ─────────────────────────────────────────────────────

    /// Start (or resume) a download on a background thread.
    ///
    /// Returns immediately; poll [`AiEngine::download_progress`].
    ///
    /// # Errors
    /// [`AiError::UnknownModel`], [`AiError::DeviceTooSmall`] if the model
    /// does not fit this device, or [`AiError::InvalidState`] if a download
    /// for this model is already running.
    pub fn start_download(&self, model_id: &str) -> Result<()> {
        let model = self.catalog.get(model_id)?.clone();
        if let Some(reason) = model.unavailable_reason(&self.config.capability) {
            return Err(AiError::DeviceTooSmall {
                id: model.id.clone(),
                required: model.min_tier.label().to_string(),
                actual: format!(
                    "{} ({reason})",
                    self.config.capability.tier().label()
                ),
            });
        }

        let mut jobs = lock(&self.jobs);
        if jobs.get(model_id).is_some_and(|j| j.handle.is_some()) {
            return Err(AiError::InvalidState {
                expected: "no download in flight".into(),
                actual: "already downloading".into(),
            });
        }

        let cancel = CancelToken::new();
        let progress = Arc::new(Mutex::new(DownloadProgress {
            model_id: model.id.clone(),
            downloaded_bytes: self.store.partial_size(&model),
            total_bytes: model.size_bytes,
            bytes_per_sec: 0.0,
            eta_secs: None,
            state: DownloadState::Queued,
        }));

        let downloader = Downloader::new(self.store.clone(), Arc::clone(&self.source));
        let thread_cancel = cancel.clone();
        let thread_progress = Arc::clone(&progress);
        let thread_model = model.clone();

        let handle = std::thread::spawn(move || {
            let result = downloader.download(&thread_model, &thread_cancel, &mut |p| {
                *lock(&thread_progress) = p.clone();
            });
            if let Err(e) = result {
                let mut p = lock(&thread_progress);
                p.state = match e {
                    AiError::DownloadCancelled => DownloadState::Cancelled,
                    other => DownloadState::Failed {
                        reason: other.to_string(),
                    },
                };
            }
        });

        jobs.insert(
            model.id,
            DownloadJob {
                cancel,
                progress,
                handle: Some(handle),
            },
        );
        Ok(())
    }

    /// Current progress for a download, if one has been started this session.
    #[must_use]
    pub fn download_progress(&self, model_id: &str) -> Option<DownloadProgress> {
        let mut jobs = lock(&self.jobs);
        let job = jobs.get_mut(model_id)?;
        // Reap a finished thread so a later `start_download` is allowed.
        if job.handle.as_ref().is_some_and(JoinHandle::is_finished) {
            if let Some(h) = job.handle.take() {
                let _ = h.join();
            }
        }
        let snapshot = lock(&job.progress).clone();
        Some(snapshot)
    }

    /// Ask a download to stop. The partial file is kept so it can resume.
    pub fn cancel_download(&self, model_id: &str) {
        if let Some(job) = lock(&self.jobs).get(model_id) {
            job.cancel.cancel();
        }
    }

    /// Cancel and wait for every in-flight download. Call before shutdown.
    pub fn shutdown_downloads(&self) {
        let mut jobs = lock(&self.jobs);
        for job in jobs.values_mut() {
            job.cancel.cancel();
        }
        for job in jobs.values_mut() {
            if let Some(h) = job.handle.take() {
                let _ = h.join();
            }
        }
    }

    /// Throw away a partial download.
    ///
    /// # Errors
    /// [`AiError::UnknownModel`] or [`AiError::Io`].
    pub fn discard_partial(&self, model_id: &str) -> Result<()> {
        let model = self.catalog.get(model_id)?;
        self.store.discard_partial(model)
    }

    /// Delete an installed model. Returns bytes reclaimed.
    ///
    /// # Errors
    /// [`AiError::UnknownModel`] or [`AiError::Io`].
    pub fn delete_model(&self, model_id: &str) -> Result<u64> {
        let model = self.catalog.get(model_id)?;
        self.cancel_download(model_id);
        self.store.delete(model)
    }

    /// Re-hash an installed model against the catalogue.
    ///
    /// # Errors
    /// [`AiError::UnknownModel`], [`AiError::ModelNotInstalled`],
    /// [`AiError::IntegrityMismatch`], or [`AiError::Config`] while the
    /// catalogue still carries a placeholder digest.
    pub fn verify_model(&self, model_id: &str) -> Result<()> {
        let model = self.catalog.get(model_id)?;
        self.store.verify(model)
    }

    // ── inference backends ──────────────────────────────────────────────────

    /// Load an installed STT model.
    ///
    /// # Errors
    /// [`AiError::UnknownModel`], [`AiError::ModelNotInstalled`],
    /// [`AiError::Config`] if the model is not an STT model, or
    /// [`AiError::FeatureNotEnabled`] when the backend it needs was compiled
    /// out — which is the default build.
    #[allow(unused_variables)]
    pub fn transcriber(&self, model_id: &str) -> Result<Arc<dyn Transcriber>> {
        let model = self.catalog.get(model_id)?;
        if model.kind != ModelKind::Stt {
            return Err(AiError::Config(format!("{model_id} is not an STT model")));
        }
        let installed = self.store.installed_model(model)?;

        match model.runtime {
            #[cfg(feature = "stt-parakeet")]
            crate::models::ModelRuntime::SherpaOnnx => {
                Ok(Arc::new(crate::stt::parakeet::ParakeetTranscriber::load(
                    model,
                    &installed.path,
                    &self.config.capability,
                )?))
            }
            #[cfg(feature = "stt-whisper")]
            crate::models::ModelRuntime::WhisperCpp => {
                Ok(Arc::new(crate::stt::whisper::WhisperTranscriber::load(
                    model,
                    &installed.path,
                    &self.config.capability,
                )?))
            }
            other => Err(AiError::FeatureNotEnabled(other.feature_name().to_string())),
        }
    }

    /// Load an installed summarization model.
    ///
    /// # Errors
    /// As [`AiEngine::transcriber`].
    #[allow(unused_variables)]
    pub fn summarizer(&self, model_id: &str) -> Result<Arc<dyn Summarizer>> {
        let model = self.catalog.get(model_id)?;
        if model.kind != ModelKind::Llm {
            return Err(AiError::Config(format!("{model_id} is not an LLM model")));
        }
        let installed = self.store.installed_model(model)?;

        match model.runtime {
            #[cfg(feature = "llm-llama")]
            crate::models::ModelRuntime::LlamaCpp => {
                Ok(Arc::new(crate::llm::llama::LlamaSummarizer::load(
                    model,
                    &installed.path,
                    &self.config.capability,
                )?))
            }
            other => Err(AiError::FeatureNotEnabled(other.feature_name().to_string())),
        }
    }

    // ── per-call ────────────────────────────────────────────────────────────

    /// Begin capturing a call. The returned session is also retained here so
    /// the host can find it again by call id.
    ///
    /// # Errors
    /// [`AiError::Config`] for an empty call id, or [`AiError::InvalidState`]
    /// if a session for this call is already open.
    pub fn begin_call(&self, config: SessionConfig) -> Result<Arc<CallSession>> {
        let call_id = config.call_id.clone();
        let mut sessions = lock(&self.sessions);
        if sessions.contains_key(&call_id) {
            return Err(AiError::InvalidState {
                expected: "no session for this call".into(),
                actual: "session already open".into(),
            });
        }
        let session = CallSession::start(config)?;
        sessions.insert(call_id, Arc::clone(&session));
        Ok(session)
    }

    /// The session for a call, if one is open.
    #[must_use]
    pub fn session(&self, call_id: &str) -> Option<Arc<CallSession>> {
        lock(&self.sessions).get(call_id).cloned()
    }

    /// Seal a call's capture and hand back the session for analysis.
    ///
    /// # Errors
    /// [`AiError::UnknownModel`]-shaped [`AiError::InvalidState`] if there is
    /// no session for this call, or whatever sealing returns.
    pub fn end_call(&self, call_id: &str) -> Result<Arc<CallSession>> {
        let session = self.session(call_id).ok_or_else(|| AiError::InvalidState {
            expected: "an open session".into(),
            actual: format!("no session for call {call_id}"),
        })?;
        session.stop()?;
        Ok(session)
    }

    /// Forget a call's session, freeing anything it still holds.
    pub fn forget_call(&self, call_id: &str) {
        // Bound the guard to a statement of its own: holding it across the
        // `if let` body would keep the sessions map locked during `discard`.
        let removed = lock(&self.sessions).remove(call_id);
        if let Some(session) = removed {
            session.discard();
        }
    }

    /// Call ids with an open session.
    #[must_use]
    pub fn open_calls(&self) -> Vec<String> {
        lock(&self.sessions).keys().cloned().collect()
    }
}

impl Drop for AiEngine {
    fn drop(&mut self) {
        self.shutdown_downloads();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::capability::Platform;
    use crate::models::download::fake::FakeSource;
    use crate::types::Speaker;

    fn engine(cap: DeviceCapability) -> (tempfile::TempDir, AiEngine) {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = FakeSource::new(vec![0_u8; 128]);
        let e = AiEngine::with_source(AiConfig::new(dir.path(), cap), source).expect("engine");
        (dir, e)
    }

    fn phone(ram_mb: u32) -> DeviceCapability {
        DeviceCapability {
            platform: Platform::Ios,
            total_ram_mb: ram_mb,
            available_ram_mb: ram_mb / 2,
            cpu_cores: 6,
            is_64bit: true,
            has_gpu_backend: true,
        }
    }

    #[test]
    fn the_engine_creates_its_directories_under_the_host_root() {
        let (dir, e) = engine(phone(8_192));
        assert!(dir.path().join("models").is_dir());
        assert!(dir.path().join("tmp").is_dir());
        assert_eq!(e.storage_bytes(), 0);
        assert_eq!(e.installed_models(), []);
    }

    #[test]
    fn availability_is_annotated_not_filtered() {
        let (_d, e) = engine(phone(4_096));
        let rows = e.availability();
        assert_eq!(rows.len(), e.catalog().models.len());
        assert!(rows.iter().any(|r| !r.available));
        assert!(rows
            .iter()
            .filter(|r| !r.available)
            .all(|r| r.reason.is_some()));
    }

    #[test]
    fn recommendations_respect_the_device_tier() {
        let (_d, low) = engine(phone(4_096));
        assert!(low.recommended(ModelKind::Stt).is_some());
        assert!(
            low.recommended(ModelKind::Llm).is_none(),
            "low-tier devices are offered no summarizer"
        );

        let (_d, desktop) = engine(DeviceCapability::desktop_default());
        assert!(desktop.recommended(ModelKind::Llm).is_some());
    }

    #[test]
    fn downloading_a_model_the_device_cannot_run_is_refused_up_front() {
        let (_d, e) = engine(phone(4_096));
        match e.start_download("qwen3-4b-instruct-2507-q4_k_m") {
            Err(AiError::DeviceTooSmall { id, .. }) => {
                assert_eq!(id, "qwen3-4b-instruct-2507-q4_k_m");
            }
            other => panic!("expected DeviceTooSmall, got {other:?}"),
        }
    }

    #[test]
    fn downloading_an_unknown_model_is_an_error() {
        let (_d, e) = engine(phone(8_192));
        assert!(matches!(
            e.start_download("no-such-model"),
            Err(AiError::UnknownModel(_))
        ));
        assert!(matches!(
            e.delete_model("no-such-model"),
            Err(AiError::UnknownModel(_))
        ));
        assert!(matches!(
            e.verify_model("no-such-model"),
            Err(AiError::UnknownModel(_))
        ));
    }

    #[test]
    fn a_download_that_does_not_match_the_catalogue_fails_and_installs_nothing() {
        let (_d, e) = engine(phone(8_192));
        e.start_download("whisper-tiny-q5_1").expect("start");

        // Poll until the worker finishes.
        let mut progress = None;
        for _ in 0..200 {
            progress = e.download_progress("whisper-tiny-q5_1");
            if progress.as_ref().is_some_and(DownloadProgress::is_terminal) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let p = progress.expect("progress");
        // The stub source serves a short body, so this exercises the path a
        // truncating proxy or a swapped CDN object would take: the download is
        // rejected and nothing lands in the models directory.
        match p.state {
            DownloadState::Failed { reason } => assert!(
                reason.contains("bytes") || reason.contains("integrity"),
                "unexpected failure reason: {reason}"
            ),
            other => panic!("expected Failed, got {other:?}"),
        }
        assert!(!e.installed_models().iter().any(|m| m.id == "whisper-tiny-q5_1"));
    }

    #[test]
    fn progress_for_a_model_that_was_never_started_is_none() {
        let (_d, e) = engine(phone(8_192));
        assert!(e.download_progress("whisper-tiny-q5_1").is_none());
        // Cancelling something that is not running is a no-op, not a panic.
        e.cancel_download("whisper-tiny-q5_1");
    }

    #[test]
    fn inference_backends_report_which_feature_is_missing() {
        let (_d, e) = engine(DeviceCapability::desktop_default());
        let model = e.descriptor("whisper-base-q5_1").expect("model");
        // Not installed yet.
        assert!(matches!(
            e.transcriber("whisper-base-q5_1"),
            Err(AiError::ModelNotInstalled(_))
        ));

        std::fs::write(e.store().path_for(model), b"weights").expect("write");
        let result = e.transcriber("whisper-base-q5_1");
        if cfg!(feature = "stt-whisper") {
            // With the real backend compiled in, a 7-byte file must fail to
            // load rather than silently succeed.
            assert!(result.is_err());
        } else {
            match result.err() {
                Some(AiError::FeatureNotEnabled(f)) => assert_eq!(f, "stt-whisper"),
                other => panic!("expected FeatureNotEnabled, got {other:?}"),
            }
        }
    }

    #[test]
    fn asking_for_the_wrong_kind_of_model_is_rejected() {
        let (_d, e) = engine(DeviceCapability::desktop_default());
        assert!(matches!(
            e.summarizer("whisper-base-q5_1"),
            Err(AiError::Config(_))
        ));
        assert!(matches!(
            e.transcriber("qwen3-1.7b-instruct-q4_k_m"),
            Err(AiError::Config(_))
        ));
    }

    #[test]
    fn sessions_are_keyed_by_call_id_and_cannot_be_opened_twice() {
        let (_d, e) = engine(phone(8_192));
        let s = e.begin_call(SessionConfig::new("call-a")).expect("begin");
        assert_eq!(s.call_id(), "call-a");
        assert_eq!(e.open_calls(), vec!["call-a".to_string()]);

        assert!(matches!(
            e.begin_call(SessionConfig::new("call-a")),
            Err(AiError::InvalidState { .. })
        ));

        s.push_local(&[100; 160], 8_000).expect("push");
        let ended = e.end_call("call-a").expect("end");
        assert!(ended.is_stopped());

        e.forget_call("call-a");
        assert_eq!(e.open_calls(), [] as [String; 0]);
        assert!(e.session("call-a").is_none());
    }

    #[test]
    fn ending_a_call_that_was_never_started_is_an_error() {
        let (_d, e) = engine(phone(8_192));
        assert!(matches!(
            e.end_call("ghost"),
            Err(AiError::InvalidState { .. })
        ));
    }

    #[test]
    fn a_full_call_runs_end_to_end_through_the_facade() {
        use crate::llm::mock::MockSummarizer;
        use crate::llm::prompt::CallContext;
        use crate::llm::pipeline::SummaryConfig;
        use crate::stt::MockTranscriber;
        use crate::util::ms_to_samples;

        let (_d, e) = engine(DeviceCapability::desktop_default());
        let session = e.begin_call(SessionConfig::new("call-e2e")).expect("begin");

        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let speech: Vec<i16> = (0..ms_to_samples(12_000, 8_000))
            .map(|i| {
                let t = i as f32 / 8_000.0;
                (0.6 * (2.0 * std::f32::consts::PI * 300.0 * t).sin() * 32_767.0) as i16
            })
            .collect();
        let quiet = vec![0_i16; speech.len()];

        for (speaker_local, block) in [(true, &speech), (false, &speech), (true, &speech)] {
            let (l, r) = if speaker_local {
                (block.as_slice(), quiet.as_slice())
            } else {
                (quiet.as_slice(), block.as_slice())
            };
            for (lf, rf) in l.chunks(160).zip(r.chunks(160)) {
                session.push_local(lf, 8_000).expect("push");
                session.push_remote(rf, 8_000).expect("push");
            }
            session.pump().expect("pump");
        }

        e.end_call("call-e2e").expect("end");

        let transcript = session
            .transcribe(&MockTranscriber::new())
            .expect("transcribe");
        assert_ne!(transcript.segments, []);
        assert!(transcript
            .segments
            .iter()
            .any(|s| s.speaker == Speaker::Remote));

        session
            .summarize(
                &MockSummarizer::new(),
                &CallContext::default(),
                &SummaryConfig::default(),
            )
            .expect("summarize");

        let insight = session.insight();
        assert!(insight.transcript.is_some());
        assert!(insight.summary.is_some());
    }
}

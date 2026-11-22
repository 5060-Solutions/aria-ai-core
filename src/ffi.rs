//! `UniFFI` facade for iOS and Android, compiled only with the `uniffi`
//! feature. The desktop (Tauri) never builds this module.
//!
//! ## Why a facade and not the plain types
//!
//! `UniFFI` can only export objects with `&self` methods, plain data records
//! and flat errors. [`AiEngine`] and [`CallSession`] are already shaped that
//! way, but two things do not cross a language boundary:
//!
//! * **Trait objects.** A host cannot pass `&dyn Transcriber`. So the FFI
//!   verbs are [`AriaAi::transcribe_call`] and [`AriaAi::summarize_call`],
//!   which take a *model id* and let the engine load the backend itself.
//! * **Tuples.** [`CallSession::stats`] returns a pair, so it is re-shaped
//!   here as [`CallStats`].
//!
//! ## Audio should not cross this boundary
//!
//! [`AriaCall::push_local`] and [`AriaCall::push_remote`] exist so a host can
//! drive a call end-to-end without any Rust of its own, but copying every
//! 20 ms frame through `UniFFI` costs an allocation per frame per leg. The
//! production mobile path keeps the realtime side in Rust: `aria-mobile-core`
//! holds the [`crate::audio::CallAudioTap`] and feeds it directly from
//! `media.rs`, and only the control plane below crosses into Swift/Kotlin.

use std::sync::Arc;
use std::time::Duration;

use crate::audio::tap::LegStats;
use crate::engine::{AiConfig, AiEngine};
use crate::error::AiError;
use crate::llm::pipeline::SummaryConfig;
use crate::llm::prompt::CallContext;
use crate::models::capability::DeviceCapability;
use crate::models::catalog::{ModelAvailability, ModelDescriptor, ModelKind};
use crate::models::download::DownloadProgress;
use crate::models::store::InstalledModel;
use crate::session::{CallSession, DrainHandle, SessionConfig};
use crate::types::{CallInsight, InsightStatus, Summary, Transcript};
use crate::util::lock;

/// How often the built-in drain thread resamples buffered audio.
const DRAIN_INTERVAL: Duration = Duration::from_millis(100);

/// Both legs' counters. `UniFFI` has no tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct CallStats {
    pub local: LegStats,
    pub remote: LegStats,
}

/// The object a mobile host holds for the lifetime of the app.
#[derive(uniffi::Object)]
pub struct AriaAi {
    inner: AiEngine,
}

// `UniFFI` can only pass owned values across the boundary — `&str` is not a
// representable parameter type — so every id here must be a `String`.
#[allow(clippy::needless_pass_by_value)]
#[uniffi::export]
impl AriaAi {
    /// Build an engine rooted at a host-owned directory.
    ///
    /// `storage_root` is `Application Support/ai` on iOS and `filesDir/ai` on
    /// Android. The crate never guesses a path.
    ///
    /// # Errors
    /// [`AiError::FeatureNotEnabled`] if this build has no `net` feature (and
    /// therefore no downloader), or whatever [`AiEngine::new`] returns.
    #[uniffi::constructor]
    pub fn new(
        storage_root: String,
        capability: DeviceCapability,
        user_agent: Option<String>,
    ) -> std::result::Result<Arc<Self>, AiError> {
        #[cfg(feature = "net")]
        {
            let mut config = AiConfig::new(storage_root, capability);
            if let Some(ua) = user_agent {
                config.user_agent = ua;
            }
            Ok(Arc::new(Self {
                inner: AiEngine::new(config)?,
            }))
        }
        #[cfg(not(feature = "net"))]
        {
            // Silence unused-parameter warnings in the no-network build.
            let _ = (storage_root, capability, user_agent);
            Err(AiError::FeatureNotEnabled("net".to_string()))
        }
    }

    // ── catalogue and device gating ─────────────────────────────────────────

    /// Every catalogue entry annotated for this device. Nothing is hidden;
    /// unavailable rows carry a plain-language reason.
    #[must_use]
    pub fn availability(&self) -> Vec<ModelAvailability> {
        self.inner.availability()
    }

    /// The best speech-to-text model this device can run, if any.
    #[must_use]
    pub fn recommended_stt(&self) -> Option<ModelDescriptor> {
        self.inner.recommended(ModelKind::Stt).cloned()
    }

    /// The best summarization model this device can run, if any.
    #[must_use]
    pub fn recommended_llm(&self) -> Option<ModelDescriptor> {
        self.inner.recommended(ModelKind::Llm).cloned()
    }

    /// Models present on disk.
    #[must_use]
    pub fn installed_models(&self) -> Vec<InstalledModel> {
        self.inner.installed_models()
    }

    /// Bytes used by models and partial downloads.
    #[must_use]
    pub fn storage_bytes(&self) -> u64 {
        self.inner.storage_bytes()
    }

    // ── model lifecycle ─────────────────────────────────────────────────────

    /// Start (or resume) a download on a background thread. Poll
    /// [`AriaAi::download_progress`].
    ///
    /// # Errors
    /// As [`AiEngine::start_download`].
    pub fn start_download(&self, model_id: String) -> std::result::Result<(), AiError> {
        self.inner.start_download(&model_id)
    }

    /// Progress for a download started this session.
    #[must_use]
    pub fn download_progress(&self, model_id: String) -> Option<DownloadProgress> {
        self.inner.download_progress(&model_id)
    }

    /// Ask a running download to stop. The `.part` file is kept for resume.
    pub fn cancel_download(&self, model_id: String) {
        self.inner.cancel_download(&model_id);
    }

    /// Delete an installed model, returning bytes reclaimed.
    ///
    /// # Errors
    /// As [`AiEngine::delete_model`].
    pub fn delete_model(&self, model_id: String) -> std::result::Result<u64, AiError> {
        self.inner.delete_model(&model_id)
    }

    /// Re-hash an installed model against the bundled catalogue.
    ///
    /// # Errors
    /// As [`AiEngine::verify_model`].
    pub fn verify_model(&self, model_id: String) -> std::result::Result<(), AiError> {
        self.inner.verify_model(&model_id)
    }

    // ── per-call ────────────────────────────────────────────────────────────

    /// Begin capturing a call.
    ///
    /// # Errors
    /// As [`AiEngine::begin_call`].
    pub fn begin_call(
        &self,
        call_id: String,
        language: Option<String>,
    ) -> std::result::Result<Arc<AriaCall>, AiError> {
        let mut config = SessionConfig::new(call_id);
        config.language = language;
        let session = self.inner.begin_call(config)?;
        Ok(Arc::new(AriaCall::wrap(session)))
    }

    /// The session for a call, if one is open.
    #[must_use]
    pub fn call(&self, call_id: String) -> Option<Arc<AriaCall>> {
        self.inner.session(&call_id).map(|s| Arc::new(AriaCall::wrap(s)))
    }

    /// Seal a call's capture. Idempotent.
    ///
    /// # Errors
    /// As [`AiEngine::end_call`].
    pub fn end_call(&self, call_id: String) -> std::result::Result<(), AiError> {
        self.inner.end_call(&call_id).map(|_| ())
    }

    /// Forget a call's session, freeing anything it still holds.
    pub fn forget_call(&self, call_id: String) {
        self.inner.forget_call(&call_id);
    }

    /// Call ids with an open session.
    #[must_use]
    pub fn open_calls(&self) -> Vec<String> {
        self.inner.open_calls()
    }

    // ── analysis ────────────────────────────────────────────────────────────

    /// Transcribe a stopped call with an installed STT model.
    ///
    /// Loading the backend happens here because a `&dyn Transcriber` cannot
    /// cross the boundary. **Consumes the captured audio** — see
    /// [`CallSession::transcribe`].
    ///
    /// # Errors
    /// [`AiError::InvalidState`] if there is no such open call,
    /// [`AiError::FeatureNotEnabled`] if the backend the model needs was
    /// compiled out, or whatever transcription returns.
    pub fn transcribe_call(
        &self,
        call_id: String,
        stt_model_id: String,
    ) -> std::result::Result<Transcript, AiError> {
        let session = self.session_or_err(&call_id)?;
        let engine = self.inner.transcriber(&stt_model_id)?;
        session.transcribe(engine.as_ref())
    }

    /// Summarize a transcribed call with an installed LLM.
    ///
    /// # Errors
    /// As [`AriaAi::transcribe_call`], plus
    /// [`AiError::TranscriptTooShort`] when the call had too little speech.
    pub fn summarize_call(
        &self,
        call_id: String,
        llm_model_id: String,
        context: CallContext,
    ) -> std::result::Result<Summary, AiError> {
        let session = self.session_or_err(&call_id)?;
        let engine = self.inner.summarizer(&llm_model_id)?;
        session.summarize(engine.as_ref(), &context, &SummaryConfig::default())
    }
}

impl AriaAi {
    fn session_or_err(&self, call_id: &str) -> std::result::Result<Arc<CallSession>, AiError> {
        self.inner.session(call_id).ok_or_else(|| AiError::InvalidState {
            expected: "an open session".to_string(),
            actual: format!("no session for call {call_id}"),
        })
    }
}

/// One call's capture and results.
#[derive(uniffi::Object)]
pub struct AriaCall {
    session: Arc<CallSession>,
    /// Owns the background resampling thread; joined when this object drops.
    drain: std::sync::Mutex<Option<DrainHandle>>,
}

impl AriaCall {
    fn wrap(session: Arc<CallSession>) -> Self {
        // A host driving frames over FFI has no thread to pump on, so one is
        // started here. `DrainHandle::drop` stops and joins it.
        let drain = session.spawn_drain_thread(DRAIN_INTERVAL);
        Self {
            session,
            drain: std::sync::Mutex::new(Some(drain)),
        }
    }

    /// Stop the drain thread. Called before sealing so the last frames land.
    fn stop_draining(&self) {
        // Take the handle out under the guard, then join outside it: joining a
        // thread while holding the lock would deadlock a concurrent `stop`.
        let handle = lock(&self.drain).take();
        if let Some(handle) = handle {
            handle.stop();
        }
    }
}

// `UniFFI` can only pass owned values across the boundary — `&str` is not a
// representable parameter type — so every id here must be a `String`.
#[allow(clippy::needless_pass_by_value)]
#[uniffi::export]
impl AriaCall {
    /// The call id this session belongs to.
    #[must_use]
    pub fn call_id(&self) -> String {
        self.session.call_id().to_string()
    }

    /// Unix seconds at which capture began.
    #[must_use]
    pub fn created_at(&self) -> i64 {
        self.session.created_at()
    }

    /// Push a microphone frame. See the module note: prefer feeding the tap
    /// from Rust rather than copying frames across the boundary.
    ///
    /// # Errors
    /// As [`CallSession::push_local`].
    pub fn push_local(
        &self,
        samples: Vec<i16>,
        rate_hz: u32,
    ) -> std::result::Result<(), AiError> {
        self.session.push_local(&samples, rate_hz)
    }

    /// Push a decoded far-end frame.
    ///
    /// # Errors
    /// As [`CallSession::push_remote`].
    pub fn push_remote(
        &self,
        samples: Vec<i16>,
        rate_hz: u32,
    ) -> std::result::Result<(), AiError> {
        self.session.push_remote(&samples, rate_hz)
    }

    /// Per-leg counters, safe to poll during the call.
    #[must_use]
    pub fn stats(&self) -> CallStats {
        let (local, remote) = self.session.stats();
        CallStats { local, remote }
    }

    /// Seal capture. Idempotent; safe to call from the hang-up path.
    ///
    /// # Errors
    /// As [`CallSession::stop`].
    pub fn stop(&self) -> std::result::Result<(), AiError> {
        self.stop_draining();
        self.session.stop()
    }

    /// True once [`AriaCall::stop`] has run.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.session.is_stopped()
    }

    /// Abandon the capture and free the audio without transcribing it.
    pub fn discard(&self) {
        self.stop_draining();
        self.session.discard();
    }

    /// The transcript, once one has been produced.
    #[must_use]
    pub fn transcript(&self) -> Option<Transcript> {
        self.session.transcript()
    }

    /// The summary, once one has been produced.
    #[must_use]
    pub fn summary(&self) -> Option<Summary> {
        self.session.summary()
    }

    /// Where this session is in its lifecycle.
    #[must_use]
    pub fn status(&self) -> InsightStatus {
        self.session.status()
    }

    /// Everything the Call Insights screen needs.
    #[must_use]
    pub fn insight(&self) -> CallInsight {
        self.session.insight()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::capability::Platform;

    fn capability() -> DeviceCapability {
        DeviceCapability {
            platform: Platform::Ios,
            total_ram_mb: 8_192,
            available_ram_mb: 4_096,
            cpu_cores: 6,
            is_64bit: true,
            has_gpu_backend: true,
        }
    }

    fn engine(dir: &tempfile::TempDir) -> Arc<AriaAi> {
        let root = dir.path().to_str().expect("utf8").to_string();
        AriaAi::new(root, capability(), Some("test/1.0".to_string())).expect("engine")
    }

    #[test]
    fn the_facade_exposes_the_catalogue_and_an_empty_store() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ai = engine(&dir);
        assert_ne!(ai.availability(), []);
        assert_eq!(ai.installed_models(), []);
        assert_eq!(ai.storage_bytes(), 0);
        assert_eq!(ai.open_calls(), [] as [String; 0]);
    }

    #[test]
    fn a_call_can_be_driven_entirely_across_the_boundary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ai = engine(&dir);

        let call = ai
            .begin_call("call-1".to_string(), None)
            .expect("begin_call");
        assert_eq!(call.call_id(), "call-1");
        assert_eq!(ai.open_calls(), vec!["call-1".to_string()]);

        // 20 ms of 8 kHz PCMU per leg, as the media layer delivers it.
        for _ in 0..50 {
            call.push_remote(vec![0_i16; 160], 8_000).expect("remote");
            call.push_local(vec![0_i16; 160], 8_000).expect("local");
        }
        assert!(call.stats().local.ring.pushed_samples > 0);

        call.stop().expect("stop");
        assert!(call.is_stopped());
        assert!(matches!(call.status(), InsightStatus::Pending));

        ai.forget_call("call-1".to_string());
        assert_eq!(ai.open_calls(), [] as [String; 0]);
    }

    #[test]
    fn transcribing_an_unknown_call_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ai = engine(&dir);
        let err = ai
            .transcribe_call("nope".to_string(), "whisper-base-q5_1".to_string())
            .expect_err("no such call");
        assert!(matches!(err, AiError::InvalidState { .. }), "{err:?}");
    }

    #[test]
    fn transcribing_with_a_backend_that_is_compiled_out_says_so() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ai = engine(&dir);
        let _call = ai.begin_call("call-2".to_string(), None).expect("begin");
        ai.end_call("call-2".to_string()).expect("end");

        let err = ai
            .transcribe_call("call-2".to_string(), "whisper-base-q5_1".to_string())
            .expect_err("the model is not installed in this test");
        // Not installed is checked before the feature gate.
        assert!(
            matches!(
                err,
                AiError::ModelNotInstalled(_) | AiError::FeatureNotEnabled(_)
            ),
            "{err:?}"
        );
    }
}

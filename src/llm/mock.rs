//! A deterministic [`Summarizer`] for tests and for building the UI before
//! any weights exist.
//!
//! It records every prompt it is given, which is how the chunking tests
//! assert that a long call really did produce N map calls and one reduce.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::error::{AiError, Result};
use crate::llm::{GenerationRequest, Summarizer};
use crate::util::lock;

/// A canned reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockResponse {
    /// Returned verbatim. Use non-JSON text to exercise the retry and
    /// degraded-fallback paths.
    Text(String),
    /// Returned as [`AiError::Summarization`].
    Error(String),
}

impl MockResponse {
    /// Convenience for `Text`.
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Convenience for `Error`.
    #[must_use]
    pub fn error(s: impl Into<String>) -> Self {
        Self::Error(s.into())
    }
}

/// The reply used once any scripted queue is exhausted.
pub const CANNED_SUMMARY_JSON: &str = r#"{
  "headline": "Rescheduled the Q3 review and agreed on scope",
  "abstract": "Sarah asked to move the Q3 review to the following Tuesday because of a customer escalation. Budget sign-off is still pending from finance.",
  "key_points": ["Q3 review moved to Tuesday the 12th", "Northwind escalation is the blocker"],
  "action_items": [
    {"owner": "me", "text": "Send the revised deck by Friday"},
    {"owner": "them", "text": "Confirm the room booking"}
  ],
  "topics": ["Q3 review", "scheduling"]
}"#;

/// Deterministic stand-in for a real LLM.
#[derive(Debug)]
pub struct MockSummarizer {
    model_id: String,
    context_tokens: usize,
    queue: Mutex<VecDeque<MockResponse>>,
    fallback_response: MockResponse,
    prompts: Mutex<Vec<GenerationRequest>>,
}

impl Default for MockSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MockSummarizer {
    /// Always returns a valid canned summary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_id: "mock-llm".into(),
            context_tokens: 4_096,
            queue: Mutex::new(VecDeque::new()),
            fallback_response: MockResponse::Text(CANNED_SUMMARY_JSON.into()),
            prompts: Mutex::new(Vec::new()),
        }
    }

    /// Returns `responses` in order, then the canned summary.
    #[must_use]
    pub fn with_responses<I: IntoIterator<Item = MockResponse>>(responses: I) -> Self {
        let mut m = Self::new();
        m.queue = Mutex::new(responses.into_iter().collect());
        m
    }

    /// Fails every call.
    #[must_use]
    pub fn failing(reason: impl Into<String>) -> Self {
        Self {
            fallback_response: MockResponse::Error(reason.into()),
            ..Self::new()
        }
    }

    /// Returns prose that is never valid JSON, for the degraded path.
    #[must_use]
    pub fn always_prose(text: impl Into<String>) -> Self {
        Self {
            fallback_response: MockResponse::Text(text.into()),
            ..Self::new()
        }
    }

    /// Override the reported context window.
    #[must_use]
    pub const fn with_context(mut self, tokens: usize) -> Self {
        self.context_tokens = tokens;
        self
    }

    /// Override the reported model id.
    #[must_use]
    pub fn with_model_id(mut self, id: impl Into<String>) -> Self {
        self.model_id = id.into();
        self
    }

    /// Every request seen so far, in order.
    #[must_use]
    pub fn prompts(&self) -> Vec<GenerationRequest> {
        lock(&self.prompts).clone()
    }

    /// How many completions have been requested.
    #[must_use]
    pub fn calls(&self) -> usize {
        lock(&self.prompts).len()
    }
}

impl Summarizer for MockSummarizer {
    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn context_tokens(&self) -> usize {
        self.context_tokens
    }

    fn complete(&self, req: &GenerationRequest) -> Result<String> {
        lock(&self.prompts).push(req.clone());
        let response = lock(&self.queue)
            .pop_front()
            .unwrap_or_else(|| self.fallback_response.clone());
        match response {
            MockResponse::Text(t) => Ok(t),
            MockResponse::Error(e) => Err(AiError::Summarization(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> GenerationRequest {
        GenerationRequest::new("sys", "user")
    }

    #[test]
    fn the_default_mock_returns_parseable_json() {
        let m = MockSummarizer::new();
        let out = m.complete(&req()).expect("ok");
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid json");
        assert!(v.get("headline").is_some());
        assert_eq!(m.calls(), 1);
    }

    #[test]
    fn scripted_responses_are_returned_in_order_then_the_canned_one() {
        let m = MockSummarizer::with_responses([
            MockResponse::text("first"),
            MockResponse::error("boom"),
        ]);
        assert_eq!(m.complete(&req()).expect("ok"), "first");
        assert!(matches!(
            m.complete(&req()),
            Err(AiError::Summarization(_))
        ));
        assert!(m.complete(&req()).expect("ok").contains("headline"));
        assert_eq!(m.calls(), 3);
    }

    #[test]
    fn prompts_are_recorded_for_assertions() {
        let m = MockSummarizer::new();
        let r = GenerationRequest::new("S", "U").with_grammar("g").with_max_tokens(7);
        m.complete(&r).expect("ok");
        let seen = m.prompts();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].user, "U");
        assert_eq!(seen[0].grammar.as_deref(), Some("g"));
        assert_eq!(seen[0].max_tokens, 7);
    }

    #[test]
    fn a_failing_mock_never_succeeds() {
        let m = MockSummarizer::failing("no model loaded");
        assert!(matches!(
            m.complete(&req()),
            Err(AiError::Summarization(_))
        ));
    }

    #[test]
    fn context_and_model_id_are_configurable() {
        let m = MockSummarizer::new()
            .with_context(8_192)
            .with_model_id("qwen3-1.7b-instruct-q4_k_m");
        assert_eq!(m.context_tokens(), 8_192);
        assert_eq!(m.model_id(), "qwen3-1.7b-instruct-q4_k_m");
    }
}

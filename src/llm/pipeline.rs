//! The map-reduce summarization pipeline and its output validator.
//!
//! Grammar-constrained decoding fixes the *format*; nothing fixes the
//! *content*. So everything the model returns is re-validated here: clamped,
//! trimmed, de-duplicated, and — if it still is not a summary — replaced by a
//! plain-prose fallback that is explicitly marked degraded rather than shown
//! as a real summary (design spec §4.3/§8.2).

use serde::Deserialize;

use crate::error::{AiError, Result};
use crate::llm::chunking::{head_and_tail, plan_windows, ChunkConfig};
use crate::llm::prompt::{
    fallback_user, map_user, reduce_user, single_pass_user, CallContext, MAP_SYSTEM_PROMPT,
    SUMMARY_GRAMMAR, SYSTEM_PROMPT,
};
use crate::llm::{GenerationRequest, Summarizer};
use crate::types::{ActionItem, ActionOwner, Summary, Transcript, TranscriptSegment};

/// Below this much speech, small models invent things. Show the transcript
/// and a "too short to summarize" note instead (design spec §4.4 "floor").
const DEFAULT_MIN_SPEECH_SECS: u32 = 30;

/// How to run the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SummaryConfig {
    /// Windowing. Leave `context_tokens` at 0 to adopt the engine's.
    pub chunk: ChunkConfig,
    pub min_speech_secs: u32,
    pub max_key_points: u32,
    pub max_action_items: u32,
    pub max_topics: u32,
    pub max_item_chars: u32,
    pub max_headline_chars: u32,
    /// Seconds of raw transcript from each end of the call replayed into the
    /// reduce prompt.
    pub edge_context_secs: u32,
    /// Token cap for a map-step generation.
    pub map_max_tokens: u32,
    /// Send the GBNF grammar. Engines that cannot use it ignore it.
    pub use_grammar: bool,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            chunk: ChunkConfig {
                context_tokens: 0,
                ..ChunkConfig::default()
            },
            min_speech_secs: DEFAULT_MIN_SPEECH_SECS,
            max_key_points: 7,
            max_action_items: 10,
            max_topics: 8,
            max_item_chars: 200,
            max_headline_chars: 80,
            edge_context_secs: 60,
            map_max_tokens: 256,
            use_grammar: true,
        }
    }
}

/// The summary plus enough detail to explain what the pipeline did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryOutcome {
    pub summary: Summary,
    /// Number of map windows. 1 means the call fitted in a single pass.
    pub windows: usize,
    /// How many times the model had to be asked again after bad output.
    pub retries: u32,
    pub model_id: String,
}

/// Summarize a transcript.
///
/// Single pass when the call fits the context, map-reduce with overlap when
/// it does not (design spec §4.4).
///
/// # Errors
/// - [`AiError::TranscriptTooShort`] — under `min_speech_secs` of speech.
/// - [`AiError::Summarization`] — the engine failed on every attempt.
/// - [`AiError::InvalidModelOutput`] — the engine produced nothing usable,
///   including from the plain-prose fallback.
pub fn summarize(
    transcript: &Transcript,
    engine: &dyn Summarizer,
    ctx: &CallContext,
    cfg: &SummaryConfig,
) -> Result<SummaryOutcome> {
    let speech = transcript.speech_secs();
    if transcript.is_empty() || speech < cfg.min_speech_secs {
        return Err(AiError::TranscriptTooShort {
            speech_secs: speech,
            minimum_secs: cfg.min_speech_secs,
        });
    }

    let mut chunk = cfg.chunk;
    if chunk.context_tokens == 0 {
        chunk.context_tokens = engine.context_tokens();
    }

    let segments = &transcript.segments;
    let windows = plan_windows(segments, &chunk);
    let mut retries = 0;

    let (system, user) = if windows.len() <= 1 {
        (SYSTEM_PROMPT.to_string(), single_pass_user(ctx, segments))
    } else {
        let mut sections: Vec<String> = Vec::with_capacity(windows.len());
        for (i, w) in windows.iter().enumerate() {
            let req = build_request(
                MAP_SYSTEM_PROMPT,
                &map_user(i, windows.len(), w.slice(segments)),
                cfg,
                cfg.map_max_tokens,
            );
            // One bad section must not lose the whole summary.
            match engine.complete(&req) {
                Ok(text) => sections.push(text),
                Err(_) => retries += 1,
            }
        }
        if sections.is_empty() {
            return Err(AiError::Summarization(
                "every section of the call failed to summarize".into(),
            ));
        }
        let (head, tail) = head_and_tail(segments, cfg.edge_context_secs);
        (
            SYSTEM_PROMPT.to_string(),
            reduce_user(ctx, &sections, head, tail),
        )
    };

    let req = build_request(&system, &user, cfg, 0);

    // First attempt.
    let mut last_err: Option<AiError> = None;
    for attempt in 0..2 {
        if attempt > 0 {
            retries += 1;
        }
        match engine.complete(&req) {
            Ok(raw) => match parse_summary(&raw, cfg) {
                Ok(summary) => {
                    return Ok(SummaryOutcome {
                        summary,
                        windows: windows.len(),
                        retries,
                        model_id: engine.model_id(),
                    })
                }
                Err(e) => last_err = Some(e),
            },
            Err(e) => last_err = Some(e),
        }
    }

    // Structured output failed twice. Fall back to plain prose and mark it.
    retries += 1;
    let fallback = build_plain_request(ctx, segments, cfg);
    match engine.complete(&fallback) {
        Ok(prose) if !prose.trim().is_empty() => Ok(SummaryOutcome {
            summary: degraded_summary(&prose, cfg),
            windows: windows.len(),
            retries,
            model_id: engine.model_id(),
        }),
        Ok(_) => Err(last_err.unwrap_or_else(|| {
            AiError::InvalidModelOutput("model returned nothing usable".into())
        })),
        Err(e) => Err(match last_err {
            Some(AiError::InvalidModelOutput(m)) => AiError::InvalidModelOutput(format!(
                "{m}; plain-text fallback also failed: {e}"
            )),
            _ => e,
        }),
    }
}

fn build_request(
    system: &str,
    user: &str,
    cfg: &SummaryConfig,
    max_tokens: u32,
) -> GenerationRequest {
    let mut req = GenerationRequest::new(system, user);
    if max_tokens > 0 {
        req = req.with_max_tokens(max_tokens);
    }
    if cfg.use_grammar {
        req = req.with_grammar(SUMMARY_GRAMMAR);
    }
    req
}

fn build_plain_request(
    ctx: &CallContext,
    segments: &[TranscriptSegment],
    cfg: &SummaryConfig,
) -> GenerationRequest {
    // No grammar here — the point is prose.
    GenerationRequest::new(
        "You summarize telephone calls. Use only what the transcript states. \
Never invent. Plain prose only.",
        fallback_user(ctx, segments),
    )
    .with_max_tokens(cfg.map_max_tokens.max(128))
}

// ── Parsing and validation ──────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
struct RawSummary {
    #[serde(default)]
    headline: String,
    #[serde(default, rename = "abstract", alias = "summary")]
    abstract_text: String,
    #[serde(default, alias = "keyPoints")]
    key_points: Vec<String>,
    #[serde(default, alias = "actionItems")]
    action_items: Vec<RawAction>,
    #[serde(default)]
    topics: Vec<String>,
}

/// Small models sometimes emit action items as bare strings. Accept both.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawAction {
    Structured {
        #[serde(default)]
        owner: String,
        #[serde(default)]
        text: String,
    },
    Plain(String),
}

/// Parse and validate one model response.
///
/// # Errors
/// [`AiError::InvalidModelOutput`] if there is no JSON object, if it does not
/// deserialize, or if it deserializes to something with no content at all.
pub fn parse_summary(raw: &str, cfg: &SummaryConfig) -> Result<Summary> {
    let json = extract_json_object(raw)
        .ok_or_else(|| AiError::InvalidModelOutput("no JSON object in the response".into()))?;

    let parsed: RawSummary = serde_json::from_str(json)
        .map_err(|e| AiError::InvalidModelOutput(format!("{e}")))?;

    let key_points = clean_list(
        parsed.key_points,
        cfg.max_key_points as usize,
        cfg.max_item_chars as usize,
    );
    let topics = clean_list(
        parsed.topics,
        cfg.max_topics as usize,
        cfg.max_item_chars as usize,
    );
    let action_items = clean_actions(parsed.action_items, cfg);

    let abstract_text = parsed.abstract_text.trim().to_string();
    let mut headline = truncate_chars(parsed.headline.trim(), cfg.max_headline_chars as usize);

    if headline.is_empty() && abstract_text.is_empty() && key_points.is_empty() {
        return Err(AiError::InvalidModelOutput(
            "response had no headline, abstract or key points".into(),
        ));
    }
    if headline.is_empty() {
        headline = derive_headline(&abstract_text, &key_points, cfg);
    }

    Ok(Summary {
        headline,
        abstract_text,
        key_points,
        action_items,
        topics,
        degraded: false,
    })
}

/// Wrap plain prose as an explicitly degraded summary.
fn degraded_summary(prose: &str, cfg: &SummaryConfig) -> Summary {
    let text = prose.trim().to_string();
    Summary {
        headline: derive_headline(&text, &[], cfg),
        abstract_text: text,
        key_points: Vec::new(),
        action_items: Vec::new(),
        topics: Vec::new(),
        degraded: true,
    }
}

/// Take the outermost `{ … }`. Models prepend preambles and append notes;
/// both are common enough that failing on them would be a bug.
fn extract_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    raw.get(start..=end)
}

fn clean_list(items: Vec<String>, max_items: usize, max_chars: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for item in items {
        let t = item.trim();
        if t.is_empty() || t.chars().count() > max_chars {
            continue;
        }
        if out.iter().any(|e: &String| e.eq_ignore_ascii_case(t)) {
            continue;
        }
        out.push(t.to_string());
        if out.len() >= max_items {
            break;
        }
    }
    out
}

fn clean_actions(items: Vec<RawAction>, cfg: &SummaryConfig) -> Vec<ActionItem> {
    let mut out: Vec<ActionItem> = Vec::new();
    for item in items {
        let (owner, text) = match item {
            RawAction::Structured { owner, text } => (ActionOwner::from_model_str(&owner), text),
            RawAction::Plain(text) => (ActionOwner::Unassigned, text),
        };
        let t = text.trim();
        if t.is_empty() || t.chars().count() > cfg.max_item_chars as usize {
            continue;
        }
        out.push(ActionItem {
            owner,
            text: t.to_string(),
        });
        if out.len() >= cfg.max_action_items as usize {
            break;
        }
    }
    out
}

fn derive_headline(abstract_text: &str, key_points: &[String], cfg: &SummaryConfig) -> String {
    let max = cfg.max_headline_chars as usize;
    let first_sentence = abstract_text
        .split_terminator(['.', '!', '?'])
        .map(str::trim)
        .find(|s| !s.is_empty());
    if let Some(s) = first_sentence {
        return truncate_chars(s, max);
    }
    if let Some(first) = key_points.first() {
        return truncate_chars(first, max);
    }
    "Call summary".to_string()
}

/// Truncate on a character boundary, never mid-UTF-8.
fn truncate_chars(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect::<String>().trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::mock::{MockResponse, MockSummarizer, CANNED_SUMMARY_JSON};
    use crate::types::Speaker;

    fn transcript(turns: usize, words: usize, secs_per_turn: u32) -> Transcript {
        let segments: Vec<TranscriptSegment> = (0..turns)
            .map(|i| {
                let start = i as u32 * secs_per_turn * 1_000;
                TranscriptSegment {
                    speaker: if i % 2 == 0 {
                        Speaker::Remote
                    } else {
                        Speaker::Local
                    },
                    start_ms: start,
                    end_ms: start + secs_per_turn * 1_000,
                    text: vec!["word"; words].join(" "),
                    confidence: 1.0,
                }
            })
            .collect();
        Transcript {
            call_id: "call-1".into(),
            language: "en".into(),
            duration_ms: segments.last().map_or(0, |s| s.end_ms),
            segments,
            narrowband: true,
            lost_audio: false,
        }
    }

    fn ctx() -> CallContext {
        CallContext {
            remote_label: "Sarah Chen".into(),
            date_label: "Today".into(),
            duration_secs: 512,
        }
    }

    #[test]
    fn a_call_with_too_little_speech_is_refused_rather_than_invented() {
        let t = transcript(2, 5, 4);
        let m = MockSummarizer::new();
        match summarize(&t, &m, &ctx(), &SummaryConfig::default()) {
            Err(AiError::TranscriptTooShort {
                speech_secs,
                minimum_secs,
            }) => {
                assert!(speech_secs < minimum_secs);
                assert_eq!(minimum_secs, 30);
            }
            other => panic!("expected TranscriptTooShort, got {other:?}"),
        }
        assert_eq!(m.calls(), 0, "the model must not be asked at all");
    }

    #[test]
    fn an_empty_transcript_is_refused() {
        let mut t = transcript(20, 5, 4);
        for s in &mut t.segments {
            s.text = "   ".into();
        }
        let m = MockSummarizer::new();
        assert!(matches!(
            summarize(&t, &m, &ctx(), &SummaryConfig::default()),
            Err(AiError::TranscriptTooShort { .. })
        ));
    }

    #[test]
    fn a_short_call_is_one_pass_and_one_model_call() {
        let t = transcript(20, 12, 4);
        let m = MockSummarizer::new();
        let out = summarize(&t, &m, &ctx(), &SummaryConfig::default()).expect("summary");

        assert_eq!(out.windows, 1);
        assert_eq!(out.retries, 0);
        assert_eq!(m.calls(), 1);
        assert_eq!(out.model_id, "mock-llm");
        assert!(!out.summary.degraded);
        assert_eq!(out.summary.key_points.len(), 2);
        assert_eq!(out.summary.action_items.len(), 2);
        assert_eq!(out.summary.action_items[0].owner, ActionOwner::Local);
        assert_eq!(out.summary.action_items[1].owner, ActionOwner::Remote);
    }

    #[test]
    fn the_single_pass_prompt_carries_the_grammar_and_the_transcript() {
        let t = transcript(20, 12, 4);
        let m = MockSummarizer::new();
        summarize(&t, &m, &ctx(), &SummaryConfig::default()).expect("summary");
        let p = m.prompts();
        assert_eq!(p.len(), 1);
        assert!(p[0].grammar.is_some());
        assert!(p[0].system.contains("Never infer or invent"));
        assert!(p[0].user.contains("Sarah Chen"));
        assert!(p[0].user.contains("[00:00] Them:"));
    }

    #[test]
    fn a_long_call_becomes_map_windows_plus_one_reduce() {
        // ~400 turns of 25 words: well past a 4k context.
        let t = transcript(400, 25, 4);
        let m = MockSummarizer::new().with_context(4_096);
        let out = summarize(&t, &m, &ctx(), &SummaryConfig::default()).expect("summary");

        assert!(out.windows > 1, "expected map-reduce, got {} window", out.windows);
        assert_eq!(m.calls(), out.windows + 1, "one call per window plus reduce");

        let prompts = m.prompts();
        let (maps, reduce) = prompts.split_at(out.windows);
        assert!(maps.iter().enumerate().all(|(i, p)| p
            .user
            .starts_with(&format!("Section {} of {}", i + 1, out.windows))));
        assert!(maps.iter().all(|p| p.max_tokens == 256));
        assert!(reduce[0].user.contains("SECTION SUMMARIES"));
        assert!(reduce[0].user.contains("OPENING OF THE CALL"));
        assert!(reduce[0].user.contains("CLOSE OF THE CALL"));
    }

    #[test]
    fn the_context_window_comes_from_the_engine_when_unset() {
        let t = transcript(400, 25, 4);
        let wide = MockSummarizer::new().with_context(128_000);
        let out = summarize(&t, &wide, &ctx(), &SummaryConfig::default()).expect("summary");
        assert_eq!(out.windows, 1, "a huge context should not need chunking");
        assert_eq!(wide.calls(), 1);
    }

    #[test]
    fn one_failed_map_section_does_not_lose_the_summary() {
        let t = transcript(400, 25, 4);
        let m = MockSummarizer::with_responses([MockResponse::error("section blew up")])
            .with_context(4_096);
        let out = summarize(&t, &m, &ctx(), &SummaryConfig::default()).expect("summary");
        assert!(out.retries >= 1);
        assert!(!out.summary.headline.is_empty());
    }

    #[test]
    fn every_map_section_failing_is_an_error() {
        let t = transcript(400, 25, 4);
        let m = MockSummarizer::failing("model unloadable").with_context(4_096);
        assert!(matches!(
            summarize(&t, &m, &ctx(), &SummaryConfig::default()),
            Err(AiError::Summarization(_))
        ));
    }

    #[test]
    fn malformed_output_is_retried_once_and_then_succeeds() {
        let t = transcript(20, 12, 4);
        let m = MockSummarizer::with_responses([MockResponse::text("I'm sorry, I can't do that")]);
        let out = summarize(&t, &m, &ctx(), &SummaryConfig::default()).expect("summary");
        assert_eq!(out.retries, 1);
        assert_eq!(m.calls(), 2);
        assert!(!out.summary.degraded);
    }

    #[test]
    fn persistent_bad_json_falls_back_to_prose_marked_degraded() {
        let t = transcript(20, 12, 4);
        let m = MockSummarizer::always_prose(
            "Sarah called to move the review. Budget sign-off is still open.",
        );
        let out = summarize(&t, &m, &ctx(), &SummaryConfig::default()).expect("summary");

        assert!(out.summary.degraded, "a prose fallback must be marked");
        assert!(out.summary.key_points.is_empty());
        assert!(out.summary.action_items.is_empty());
        assert_eq!(out.summary.headline, "Sarah called to move the review");
        assert!(out.summary.abstract_text.contains("Budget sign-off"));
        assert_eq!(m.calls(), 3, "two structured attempts plus the fallback");
        // The fallback must not constrain decoding to JSON.
        assert!(m.prompts()[2].grammar.is_none());
    }

    #[test]
    fn a_failure_on_every_path_surfaces_as_an_error() {
        let t = transcript(20, 12, 4);
        let m = MockSummarizer::with_responses([
            MockResponse::text("nope"),
            MockResponse::text("still nope"),
            MockResponse::error("engine died"),
        ]);
        assert!(summarize(&t, &m, &ctx(), &SummaryConfig::default()).is_err());
    }

    // ── validator ──────────────────────────────────────────────────────────

    #[test]
    fn a_preamble_and_trailing_notes_around_the_json_are_tolerated() {
        let raw = format!("Sure! Here is the summary:\n{CANNED_SUMMARY_JSON}\nHope that helps.");
        let s = parse_summary(&raw, &SummaryConfig::default()).expect("parse");
        assert_eq!(s.key_points.len(), 2);
    }

    #[test]
    fn key_points_action_items_and_topics_are_clamped() {
        let raw = serde_json::json!({
            "headline": "h",
            "abstract": "a",
            "key_points": (0..30).map(|i| format!("point {i}")).collect::<Vec<_>>(),
            "action_items": (0..30).map(|i| serde_json::json!({"owner":"me","text":format!("do {i}")})).collect::<Vec<_>>(),
            "topics": (0..30).map(|i| format!("topic {i}")).collect::<Vec<_>>(),
        })
        .to_string();
        let cfg = SummaryConfig::default();
        let s = parse_summary(&raw, &cfg).expect("parse");
        assert_eq!(s.key_points.len(), 7);
        assert_eq!(s.action_items.len(), 10);
        assert_eq!(s.topics.len(), 8);
    }

    #[test]
    fn empty_duplicate_and_overlong_items_are_dropped() {
        let long = "x".repeat(500);
        let raw = serde_json::json!({
            "headline": "h",
            "abstract": "a",
            "key_points": ["  ", "Same point", "same point", long, "Real point"],
            "action_items": [{"owner":"me","text":"  "}, {"owner":"nobody","text":"call back"}],
            "topics": [],
        })
        .to_string();
        let s = parse_summary(&raw, &SummaryConfig::default()).expect("parse");
        assert_eq!(s.key_points, vec!["Same point", "Real point"]);
        assert_eq!(s.action_items.len(), 1);
        assert_eq!(s.action_items[0].owner, ActionOwner::Unassigned);
        assert_eq!(s.action_items[0].text, "call back");
    }

    #[test]
    fn action_items_given_as_bare_strings_are_accepted() {
        let raw = r#"{"headline":"h","abstract":"a","key_points":[],
                      "action_items":["send the deck"],"topics":[]}"#;
        let s = parse_summary(raw, &SummaryConfig::default()).expect("parse");
        assert_eq!(s.action_items.len(), 1);
        assert_eq!(s.action_items[0].owner, ActionOwner::Unassigned);
    }

    #[test]
    fn a_long_headline_is_truncated_on_a_character_boundary() {
        let raw = serde_json::json!({
            "headline": "é".repeat(300),
            "abstract": "a", "key_points": [], "action_items": [], "topics": []
        })
        .to_string();
        let s = parse_summary(&raw, &SummaryConfig::default()).expect("parse");
        assert_eq!(s.headline.chars().count(), 80);
    }

    #[test]
    fn a_missing_headline_is_derived_from_the_abstract() {
        let raw = r#"{"abstract":"Sarah moved the review. Budget is open.",
                      "key_points":[],"action_items":[],"topics":[]}"#;
        let s = parse_summary(raw, &SummaryConfig::default()).expect("parse");
        assert_eq!(s.headline, "Sarah moved the review");
    }

    #[test]
    fn a_missing_headline_and_abstract_fall_back_to_a_key_point() {
        let raw = r#"{"key_points":["Rescheduled to Tuesday"],"action_items":[],"topics":[]}"#;
        let s = parse_summary(raw, &SummaryConfig::default()).expect("parse");
        assert_eq!(s.headline, "Rescheduled to Tuesday");
    }

    #[test]
    fn an_entirely_empty_object_is_not_a_summary() {
        assert!(matches!(
            parse_summary("{}", &SummaryConfig::default()),
            Err(AiError::InvalidModelOutput(_))
        ));
        assert!(matches!(
            parse_summary(
                r#"{"headline":"","abstract":"","key_points":[],"action_items":[],"topics":[]}"#,
                &SummaryConfig::default()
            ),
            Err(AiError::InvalidModelOutput(_))
        ));
    }

    #[test]
    fn non_json_and_broken_json_are_rejected_without_panicking() {
        let cfg = SummaryConfig::default();
        for raw in [
            "",
            "no braces here",
            "}{",
            "{ this is not json",
            "{\"key_points\": \"not an array\"}",
        ] {
            assert!(
                matches!(parse_summary(raw, &cfg), Err(AiError::InvalidModelOutput(_))),
                "accepted {raw:?}"
            );
        }
    }

    #[test]
    fn camel_case_field_names_are_accepted() {
        let raw = r#"{"headline":"h","summary":"a","keyPoints":["p"],
                      "actionItems":[{"owner":"them","text":"t"}],"topics":[]}"#;
        let s = parse_summary(raw, &SummaryConfig::default()).expect("parse");
        assert_eq!(s.abstract_text, "a");
        assert_eq!(s.key_points, vec!["p"]);
        assert_eq!(s.action_items[0].owner, ActionOwner::Remote);
    }
}

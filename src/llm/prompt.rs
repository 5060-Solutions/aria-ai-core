//! Prompts, the JSON schema, and the GBNF grammar (design spec §4.2/§4.3).
//!
//! Two things do the heavy lifting with a 1–4B model: a *short* instruction
//! block, and grammar-constrained decoding. The grammar removes malformed
//! JSON — the single largest failure mode at this size — and is the main
//! reason the spec picks llama.cpp over a generic runtime.
//!
//! The instructions lean hard on "do not invent", because the model research
//! is explicit that the real risk is not word error rate but a fluent,
//! confident, fabricated line in a customer's call summary.

use std::fmt::Write as _;

use crate::types::{render_segments, TranscriptSegment};

/// System prompt for the single-pass and reduce steps.
pub const SYSTEM_PROMPT: &str = "\
You summarize telephone calls. You are given a transcript with two speakers: \
\"Me\" (the phone's owner) and \"Them\" (the other party).

Rules:
- Use only information stated in the transcript. Never infer or invent.
- The transcript is machine-generated and may contain errors. If something is \
unclear or garbled, say so rather than guessing.
- An action item must be an explicit commitment or request made by a speaker. \
If there are none, return an empty list. Do not manufacture action items.
- Write in the same language as the transcript.
- Output JSON only, matching the given schema.";

/// System prompt for the map step of a long call.
pub const MAP_SYSTEM_PROMPT: &str = "\
You are summarizing one section of a longer telephone call, between two \
speakers: \"Me\" (the phone's owner) and \"Them\" (the other party).

Rules:
- Cover only this section. Do not speculate about what came before or after.
- Use only information stated in the transcript. Never infer or invent.
- An action item must be an explicit commitment or request. If there are \
none, return an empty list.
- Output JSON only, matching the given schema.";

/// The output contract, restated in the user message because small models
/// follow a schema they can see better than one they were told about.
pub const SCHEMA_HINT: &str = r#"Schema:
{
  "headline": "one line, at most 80 characters",
  "abstract": "2-4 sentences",
  "key_points": ["string", "..."],
  "action_items": [{"owner": "me" | "them" | "unassigned", "text": "string"}],
  "topics": ["string", "..."]
}"#;

/// GBNF grammar forcing syntactically valid output in the summary shape.
///
/// Passed to `llama.cpp` via `llama-cpp-2`. Engines that cannot constrain
/// decoding ignore it and rely on the Rust-side validator instead.
pub const SUMMARY_GRAMMAR: &str = r#"root   ::= "{" ws
  "\"headline\"" ws ":" ws string ws "," ws
  "\"abstract\"" ws ":" ws string ws "," ws
  "\"key_points\"" ws ":" ws strarray ws "," ws
  "\"action_items\"" ws ":" ws actions ws "," ws
  "\"topics\"" ws ":" ws strarray ws
"}" ws

strarray ::= "[" ws (string (ws "," ws string)*)? ws "]"
actions  ::= "[" ws (action (ws "," ws action)*)? ws "]"
action   ::= "{" ws "\"owner\"" ws ":" ws owner ws "," ws "\"text\"" ws ":" ws string ws "}"
owner    ::= "\"me\"" | "\"them\"" | "\"unassigned\""

string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" hex hex hex hex
hex    ::= [0-9a-fA-F]
ws     ::= [ \t\n]*
"#;

/// Call metadata rendered into the user message.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct CallContext {
    /// Contact name or number of the far end. Empty is fine.
    pub remote_label: String,
    /// Human-readable date/time. The core does not format dates — the host
    /// knows the user's locale and this crate does not carry a date library.
    pub date_label: String,
    pub duration_secs: u32,
}

impl CallContext {
    /// `Call with X on Y, duration mm:ss.`
    #[must_use]
    pub fn header(&self) -> String {
        let mins = self.duration_secs / 60;
        let secs = self.duration_secs % 60;
        let who = if self.remote_label.trim().is_empty() {
            "an unknown number".to_string()
        } else {
            self.remote_label.trim().to_string()
        };
        let when = if self.date_label.trim().is_empty() {
            String::new()
        } else {
            format!(" on {}", self.date_label.trim())
        };
        format!("Call with {who}{when}, duration {mins}:{secs:02}.")
    }
}

/// User message for a call that fits in one pass.
#[must_use]
pub fn single_pass_user(ctx: &CallContext, segments: &[TranscriptSegment]) -> String {
    format!(
        "{}\n\n{}\n\nTRANSCRIPT\n{}",
        ctx.header(),
        SCHEMA_HINT,
        render_segments(segments)
    )
}

/// User message for one map window.
#[must_use]
pub fn map_user(index: usize, total: usize, segments: &[TranscriptSegment]) -> String {
    format!(
        "Section {} of {}.\n\n{}\n\nTRANSCRIPT\n{}",
        index + 1,
        total,
        SCHEMA_HINT,
        render_segments(segments)
    )
}

/// User message for the reduce step: the section summaries, plus the raw
/// opening and closing of the call, which carry the greeting/identification
/// and the commitments (design spec §4.4).
#[must_use]
pub fn reduce_user(
    ctx: &CallContext,
    section_summaries: &[String],
    head: &[TranscriptSegment],
    tail: &[TranscriptSegment],
) -> String {
    let mut out = format!("{}\n\n{}\n\nSECTION SUMMARIES\n", ctx.header(), SCHEMA_HINT);
    for (i, s) in section_summaries.iter().enumerate() {
        // `write!` into a `String` is infallible; the result is discarded
        // rather than unwrapped because this crate must not panic.
        let _ = writeln!(out, "--- section {} ---\n{}", i + 1, s.trim());
    }
    if !head.is_empty() {
        out.push_str("\nOPENING OF THE CALL\n");
        out.push_str(&render_segments(head));
    }
    if !tail.is_empty() {
        out.push_str("\nCLOSE OF THE CALL\n");
        out.push_str(&render_segments(tail));
    }
    out
}

/// Fallback prompt used once structured output has failed. Plain prose only —
/// the result is stored as a degraded summary rather than shown as nothing.
#[must_use]
pub fn fallback_user(ctx: &CallContext, segments: &[TranscriptSegment]) -> String {
    format!(
        "{}\n\nWrite 2-4 plain sentences summarizing this call. \
No lists, no JSON, no preamble. Use only what the transcript states.\n\n\
TRANSCRIPT\n{}",
        ctx.header(),
        render_segments(segments)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Speaker;

    fn segs() -> Vec<TranscriptSegment> {
        vec![
            TranscriptSegment {
                speaker: Speaker::Remote,
                start_ms: 4_000,
                end_ms: 6_000,
                text: "Hi, it's Sarah".into(),
                confidence: 1.0,
            },
            TranscriptSegment {
                speaker: Speaker::Local,
                start_ms: 7_000,
                end_ms: 8_000,
                text: "Hello".into(),
                confidence: 1.0,
            },
        ]
    }

    #[test]
    fn the_header_formats_duration_as_minutes_and_seconds() {
        let ctx = CallContext {
            remote_label: "Sarah Chen".into(),
            date_label: "Today 2:14 PM".into(),
            duration_secs: 512,
        };
        assert_eq!(
            ctx.header(),
            "Call with Sarah Chen on Today 2:14 PM, duration 8:32."
        );
    }

    #[test]
    fn a_missing_contact_and_date_degrade_gracefully() {
        let ctx = CallContext {
            remote_label: "  ".into(),
            date_label: String::new(),
            duration_secs: 5,
        };
        assert_eq!(ctx.header(), "Call with an unknown number, duration 0:05.");
    }

    #[test]
    fn the_single_pass_prompt_carries_schema_and_transcript() {
        let p = single_pass_user(&CallContext::default(), &segs());
        assert!(p.contains("key_points"));
        assert!(p.contains("[00:04] Them: Hi, it's Sarah"));
        assert!(p.contains("[00:07] Me: Hello"));
    }

    #[test]
    fn the_map_prompt_states_which_section_it_is() {
        let p = map_user(2, 7, &segs());
        assert!(p.starts_with("Section 3 of 7."));
    }

    #[test]
    fn the_reduce_prompt_includes_sections_and_the_call_edges() {
        let s = segs();
        let p = reduce_user(
            &CallContext::default(),
            &["first".into(), "second".into()],
            &s[..1],
            &s[1..],
        );
        assert!(p.contains("--- section 1 ---"));
        assert!(p.contains("--- section 2 ---"));
        assert!(p.contains("OPENING OF THE CALL"));
        assert!(p.contains("CLOSE OF THE CALL"));
    }

    #[test]
    fn the_reduce_prompt_omits_empty_edges() {
        let p = reduce_user(&CallContext::default(), &["only".into()], &[], &[]);
        assert!(!p.contains("OPENING OF THE CALL"));
        assert!(!p.contains("CLOSE OF THE CALL"));
    }

    #[test]
    fn the_fallback_prompt_forbids_json() {
        let p = fallback_user(&CallContext::default(), &segs());
        assert!(p.contains("no JSON"));
        assert!(!p.contains("key_points"));
    }

    #[test]
    fn the_grammar_covers_every_schema_field() {
        for field in [
            "headline",
            "abstract",
            "key_points",
            "action_items",
            "topics",
            "owner",
        ] {
            assert!(SUMMARY_GRAMMAR.contains(field), "grammar missing {field}");
        }
        assert!(SUMMARY_GRAMMAR.contains("root"));
    }

    #[test]
    fn the_instructions_state_the_anti_hallucination_rules() {
        assert!(SYSTEM_PROMPT.contains("Never infer or invent"));
        assert!(SYSTEM_PROMPT.contains("Do not manufacture action items"));
        assert!(MAP_SYSTEM_PROMPT.contains("Never infer or invent"));
    }
}

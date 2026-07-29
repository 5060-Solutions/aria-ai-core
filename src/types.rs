//! Result types shared by every platform. These are the shapes the design
//! spec fixes in §2.4 so that Call Insights looks identical on iOS, Android
//! and desktop.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

/// Which call leg a piece of audio or text came from.
///
/// Attribution is free because the two legs are never summed (design spec
/// §3.2) — `Local` is the microphone, `Remote` is the decoded far end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[serde(rename_all = "lowercase")]
pub enum Speaker {
    /// The phone's owner. Rendered as "You" / "Me".
    Local,
    /// The other party. Rendered as the contact name / "Them".
    Remote,
}

impl Speaker {
    /// The label used inside the LLM prompt (design spec §4.2).
    #[must_use]
    pub const fn prompt_label(self) -> &'static str {
        match self {
            Self::Local => "Me",
            Self::Remote => "Them",
        }
    }

    /// The other leg.
    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::Local => Self::Remote,
            Self::Remote => Self::Local,
        }
    }
}

/// One attributed, timed piece of transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct TranscriptSegment {
    pub speaker: Speaker,
    pub start_ms: u32,
    pub end_ms: u32,
    pub text: String,
    /// Engine-reported confidence in `[0, 1]`. Engines that do not report one
    /// return `1.0`; do not treat this as calibrated.
    pub confidence: f32,
}

impl TranscriptSegment {
    /// Duration of this segment.
    #[must_use]
    pub const fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// The full, merged, speaker-attributed transcript for one call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct Transcript {
    pub call_id: String,
    /// BCP-47-ish language tag, detected or forced.
    pub language: String,
    pub duration_ms: u32,
    /// Sorted by `start_ms`. Overlapping segments are legal and expected —
    /// people talk over each other (design spec §3.2).
    pub segments: Vec<TranscriptSegment>,
    /// The call was carried at 8 kHz, so accuracy is materially worse. The UI
    /// must say so (design spec §3.3).
    pub narrowband: bool,
    /// Audio was dropped during capture; the transcript has holes.
    pub lost_audio: bool,
}

impl Transcript {
    /// Total wall-clock seconds covered by segments, counting overlap once.
    #[must_use]
    pub fn speech_secs(&self) -> u32 {
        let mut spans: Vec<(u32, u32)> = self
            .segments
            .iter()
            .map(|s| (s.start_ms, s.end_ms))
            .collect();
        spans.sort_unstable();
        let mut total_ms: u64 = 0;
        let mut cur: Option<(u32, u32)> = None;
        for (start, end) in spans {
            match cur {
                Some((cs, ce)) if start <= ce => cur = Some((cs, ce.max(end))),
                Some((cs, ce)) => {
                    total_ms += u64::from(ce.saturating_sub(cs));
                    cur = Some((start, end));
                }
                None => cur = Some((start, end)),
            }
        }
        if let Some((cs, ce)) = cur {
            total_ms += u64::from(ce.saturating_sub(cs));
        }
        u32::try_from(total_ms / 1000).unwrap_or(u32::MAX)
    }

    /// Render as the `[mm:ss] Speaker: text` block the summarizer prompt uses.
    #[must_use]
    pub fn to_prompt_text(&self) -> String {
        render_segments(&self.segments)
    }

    /// True when there is nothing worth showing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.iter().all(|s| s.text.trim().is_empty())
    }
}

/// Render a slice of segments in prompt form. Shared by chunking.
#[must_use]
pub fn render_segments(segments: &[TranscriptSegment]) -> String {
    let mut out = String::new();
    for seg in segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        let secs = seg.start_ms / 1000;
        // `write!` into a `String` is infallible; the result is discarded
        // rather than unwrapped because this crate must not panic.
        let _ = writeln!(
            out,
            "[{:02}:{:02}] {}: {}",
            secs / 60,
            secs % 60,
            seg.speaker.prompt_label(),
            text
        );
    }
    out
}

/// Who owns an action item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[serde(rename_all = "lowercase")]
pub enum ActionOwner {
    /// "me" in the model's JSON.
    Local,
    /// "them" in the model's JSON.
    Remote,
    /// "unassigned" in the model's JSON.
    Unassigned,
}

impl ActionOwner {
    /// Parse the model's JSON spelling. Anything unrecognised is `Unassigned`
    /// rather than an error — a small model getting the enum wrong should not
    /// throw away an otherwise good summary.
    #[must_use]
    pub fn from_model_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "me" | "local" | "you" => Self::Local,
            "them" | "remote" | "they" => Self::Remote,
            _ => Self::Unassigned,
        }
    }
}

/// One extracted commitment or request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct ActionItem {
    pub owner: ActionOwner,
    pub text: String,
}

/// The structured summary the LLM produces (design spec §4.3).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct Summary {
    /// One line, <= 80 chars.
    pub headline: String,
    /// 2–4 sentences.
    #[serde(rename = "abstract")]
    pub abstract_text: String,
    /// 3–7 points, clamped to 7.
    pub key_points: Vec<String>,
    /// Clamped to 10.
    pub action_items: Vec<ActionItem>,
    pub topics: Vec<String>,
    /// True when structured parsing failed and this is a plain-text fallback.
    /// The UI must show it as degraded rather than pretend (design spec §8.2).
    #[serde(default)]
    pub degraded: bool,
}

/// Where an insight is in its lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum InsightStatus {
    /// Captured, queued, nothing done yet.
    Pending,
    /// STT running.
    Transcribing,
    /// LLM running.
    Summarizing,
    /// Transcript present; summary present or deliberately absent.
    Complete,
    /// Transcript present, summary skipped because there was too little speech.
    TranscriptOnly { reason: String },
    /// Gave up.
    Failed { reason: String },
}

/// Everything the Call Insights screen needs for one call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct CallInsight {
    pub call_id: String,
    /// Unix seconds.
    pub created_at: i64,
    pub duration_secs: u32,
    pub transcript: Option<Transcript>,
    pub summary: Option<Summary>,
    pub stt_model_id: Option<String>,
    pub llm_model_id: Option<String>,
    pub status: InsightStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(speaker: Speaker, start_ms: u32, end_ms: u32, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            speaker,
            start_ms,
            end_ms,
            text: text.to_string(),
            confidence: 1.0,
        }
    }

    fn transcript(segments: Vec<TranscriptSegment>) -> Transcript {
        Transcript {
            call_id: "c1".into(),
            language: "en".into(),
            duration_ms: 60_000,
            segments,
            narrowband: true,
            lost_audio: false,
        }
    }

    #[test]
    fn speech_seconds_counts_overlap_once() {
        let t = transcript(vec![
            seg(Speaker::Remote, 0, 10_000, "hello"),
            seg(Speaker::Local, 5_000, 15_000, "hi"),
            seg(Speaker::Remote, 40_000, 50_000, "bye"),
        ]);
        // 0..15s plus 40..50s = 25s, not 30s.
        assert_eq!(t.speech_secs(), 25);
    }

    #[test]
    fn speech_seconds_of_an_empty_transcript_is_zero() {
        assert_eq!(transcript(vec![]).speech_secs(), 0);
    }

    #[test]
    fn prompt_rendering_uses_mm_ss_and_me_them() {
        let t = transcript(vec![
            seg(Speaker::Remote, 4_200, 6_000, "Hi, it's Sarah"),
            seg(Speaker::Local, 7_000, 8_000, "  Sure  "),
            seg(Speaker::Local, 9_000, 9_500, "   "), // blank, skipped
            seg(Speaker::Remote, 125_000, 126_000, "later"),
        ]);
        let rendered = t.to_prompt_text();
        assert_eq!(
            rendered,
            "[00:04] Them: Hi, it's Sarah\n[00:07] Me: Sure\n[02:05] Them: later\n"
        );
    }

    #[test]
    fn action_owner_parsing_never_fails() {
        assert_eq!(ActionOwner::from_model_str("me"), ActionOwner::Local);
        assert_eq!(ActionOwner::from_model_str(" THEM "), ActionOwner::Remote);
        assert_eq!(
            ActionOwner::from_model_str("the marketing team"),
            ActionOwner::Unassigned
        );
        assert_eq!(ActionOwner::from_model_str(""), ActionOwner::Unassigned);
    }

    #[test]
    fn speaker_helpers() {
        assert_eq!(Speaker::Local.other(), Speaker::Remote);
        assert_eq!(Speaker::Remote.prompt_label(), "Them");
    }

    #[test]
    fn transcript_with_only_blank_text_is_empty() {
        assert!(transcript(vec![seg(Speaker::Local, 0, 100, "   ")]).is_empty());
        assert!(!transcript(vec![seg(Speaker::Local, 0, 100, "x")]).is_empty());
    }
}

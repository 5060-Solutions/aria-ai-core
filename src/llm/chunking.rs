//! Splitting a long call into windows that fit the model's context
//! (design spec §4.4).
//!
//! Budget arithmetic from the spec: conversational speech runs ~130–170
//! words/minute of transcript for a two-party call; at ~1.35 tokens/word plus
//! timestamp and speaker overhead that is roughly **250 tokens per minute of
//! call**. With a 4096-token context and ~600 tokens reserved for the system
//! prompt, schema and output, a single pass tops out near 13–14 minutes.
//!
//! Beyond that: map-reduce with overlap, splitting preferentially at speaker
//! turns and at silences longer than three seconds.

use crate::types::TranscriptSegment;

/// Tokens of prompt scaffolding + generated output to keep free.
const DEFAULT_RESERVED_TOKENS: usize = 600;

/// Spec default: ~1,800 tokens ≈ 7 minutes of call per map window.
const DEFAULT_WINDOW_TOKENS: usize = 1_800;

/// Spec default: ~150 tokens of overlap between windows.
const DEFAULT_OVERLAP_TOKENS: usize = 150;

/// Prefer to cut at a silence at least this long.
const DEFAULT_GAP_SPLIT_MS: u32 = 3_000;

/// Once a window is this full, take the first good boundary offered.
const BOUNDARY_SEEK_FRACTION: usize = 70;

/// Per-segment overhead for `[mm:ss] Speaker: ` and the newline.
const SEGMENT_OVERHEAD_TOKENS: usize = 8;

/// How to slice a transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    /// The engine's usable context window.
    pub context_tokens: usize,
    /// Held back for system prompt, schema and generated output.
    pub reserved_tokens: usize,
    /// Target size of a map window.
    pub window_tokens: usize,
    /// Tokens of the previous window replayed into the next.
    pub overlap_tokens: usize,
    /// Silence at least this long is a preferred cut point.
    pub gap_split_ms: u32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            context_tokens: 4_096,
            reserved_tokens: DEFAULT_RESERVED_TOKENS,
            window_tokens: DEFAULT_WINDOW_TOKENS,
            overlap_tokens: DEFAULT_OVERLAP_TOKENS,
            gap_split_ms: DEFAULT_GAP_SPLIT_MS,
        }
    }
}

impl ChunkConfig {
    /// A config sized for an engine with `context_tokens` of context.
    #[must_use]
    pub fn for_context(context_tokens: usize) -> Self {
        Self {
            context_tokens,
            ..Self::default()
        }
    }

    /// Tokens of transcript that fit in one prompt.
    #[must_use]
    pub const fn single_pass_budget(&self) -> usize {
        // Always leave room for something, even with a nonsense config.
        let ctx = if self.context_tokens < 512 {
            512
        } else {
            self.context_tokens
        };
        let reserved = if self.reserved_tokens >= ctx {
            ctx / 2
        } else {
            self.reserved_tokens
        };
        ctx - reserved
    }

    /// Effective map-window size, never larger than the single-pass budget.
    #[must_use]
    pub const fn effective_window_tokens(&self) -> usize {
        let budget = self.single_pass_budget();
        let w = if self.window_tokens < 128 {
            128
        } else {
            self.window_tokens
        };
        if w > budget {
            budget
        } else {
            w
        }
    }
}

/// A contiguous run of segments to hand to the model as one prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// Inclusive index of the first segment.
    pub start_index: u32,
    /// Exclusive index of the last segment.
    pub end_index: u32,
    pub start_ms: u32,
    pub end_ms: u32,
    pub est_tokens: u32,
}

impl Window {
    /// Borrow the segments this window covers.
    #[must_use]
    pub fn slice<'a>(&self, segments: &'a [TranscriptSegment]) -> &'a [TranscriptSegment] {
        let start = self.start_index as usize;
        let end = (self.end_index as usize).min(segments.len());
        if start >= end {
            return &[];
        }
        segments.get(start..end).unwrap_or(&[])
    }
}

/// Estimated token count for free text.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    // ~1.35 tokens per word.
    words * 27 / 20
}

/// Estimated token cost of one rendered transcript line.
#[must_use]
pub fn segment_tokens(segment: &TranscriptSegment) -> usize {
    estimate_tokens(&segment.text) + SEGMENT_OVERHEAD_TOKENS
}

/// Estimated token cost of a whole transcript.
#[must_use]
pub fn transcript_tokens(segments: &[TranscriptSegment]) -> usize {
    segments.iter().map(segment_tokens).sum()
}

/// Plan the map windows for a transcript.
///
/// Returns a single window when the whole transcript fits in one prompt —
/// that is the short-circuit the spec asks for, and the common case.
/// Windows are always non-empty and always advance, so a pathological input
/// (one segment larger than the window) yields that segment alone rather than
/// looping.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn plan_windows(segments: &[TranscriptSegment], cfg: &ChunkConfig) -> Vec<Window> {
    if segments.is_empty() {
        return Vec::new();
    }

    let total = transcript_tokens(segments);
    if total <= cfg.single_pass_budget() {
        return vec![window_of(segments, 0, segments.len())];
    }

    let window_tokens = cfg.effective_window_tokens();
    let seek_from = window_tokens * BOUNDARY_SEEK_FRACTION / 100;

    let mut windows: Vec<Window> = Vec::new();
    let mut start = 0_usize;

    while start < segments.len() {
        let mut used = 0_usize;
        let mut end = start;

        while end < segments.len() {
            let Some(seg) = segments.get(end) else { break };
            let cost = segment_tokens(seg);

            // Always take at least one segment, even an oversized one.
            if end > start && used + cost > window_tokens {
                break;
            }
            used += cost;
            end += 1;

            // Past the seek threshold, cut at the first natural boundary.
            if used >= seek_from && end < segments.len() && is_good_boundary(segments, end, cfg) {
                break;
            }
        }

        // Guaranteed forward progress.
        if end == start {
            end = start + 1;
        }
        windows.push(window_of(segments, start, end));

        if end >= segments.len() {
            break;
        }
        start = backoff_for_overlap(segments, start, end, cfg.overlap_tokens);
    }

    windows
}

/// A window covering the first and last `secs` of a call — the greeting and
/// the close, which carry disproportionate signal in the reduce step
/// (design spec §4.4).
#[must_use]
pub fn head_and_tail(
    segments: &[TranscriptSegment],
    secs: u32,
) -> (&[TranscriptSegment], &[TranscriptSegment]) {
    if segments.is_empty() {
        return (&[], &[]);
    }
    let window_ms = secs.saturating_mul(1_000);
    let last_end = segments.last().map_or(0, |s| s.end_ms);

    let head_end = segments
        .iter()
        .position(|s| s.start_ms > window_ms)
        .unwrap_or(segments.len());
    let tail_start = segments
        .iter()
        .position(|s| s.end_ms >= last_end.saturating_sub(window_ms))
        .unwrap_or(0);

    let head = segments.get(..head_end).unwrap_or(&[]);
    let tail = segments.get(tail_start.max(head_end)..).unwrap_or(&[]);
    (head, tail)
}

#[allow(clippy::cast_possible_truncation)]
fn window_of(segments: &[TranscriptSegment], start: usize, end: usize) -> Window {
    let slice = segments.get(start..end).unwrap_or(&[]);
    Window {
        start_index: start as u32,
        end_index: end as u32,
        start_ms: slice.first().map_or(0, |s| s.start_ms),
        end_ms: slice.last().map_or(0, |s| s.end_ms),
        est_tokens: transcript_tokens(slice).min(u32::MAX as usize) as u32,
    }
}

/// Is index `at` a good place to cut — a long silence or a speaker change?
fn is_good_boundary(segments: &[TranscriptSegment], at: usize, cfg: &ChunkConfig) -> bool {
    let (Some(prev), Some(next)) = (segments.get(at.wrapping_sub(1)), segments.get(at)) else {
        return false;
    };
    let gap = next.start_ms.saturating_sub(prev.end_ms);
    gap >= cfg.gap_split_ms || prev.speaker != next.speaker
}

/// Walk back from `end` until ~`overlap_tokens` have been replayed, without
/// ever returning an index at or before `start`.
fn backoff_for_overlap(
    segments: &[TranscriptSegment],
    start: usize,
    end: usize,
    overlap_tokens: usize,
) -> usize {
    if overlap_tokens == 0 {
        return end;
    }
    let mut idx = end;
    let mut replayed = 0_usize;
    while idx > start + 1 {
        let Some(seg) = segments.get(idx - 1) else {
            break;
        };
        let cost = segment_tokens(seg);
        if replayed + cost > overlap_tokens {
            break;
        }
        replayed += cost;
        idx -= 1;
    }
    idx.max(start + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Speaker;

    fn seg(speaker: Speaker, start_ms: u32, end_ms: u32, words: usize) -> TranscriptSegment {
        TranscriptSegment {
            speaker,
            start_ms,
            end_ms,
            text: vec!["word"; words].join(" "),
            confidence: 1.0,
        }
    }

    /// A conversation of `turns` alternating turns, 4 s apart, `words` each.
    fn conversation(turns: usize, words: usize, gap_ms: u32) -> Vec<TranscriptSegment> {
        (0..turns)
            .map(|i| {
                let speaker = if i % 2 == 0 {
                    Speaker::Remote
                } else {
                    Speaker::Local
                };
                let start = i as u32 * gap_ms;
                seg(speaker, start, start + 2_000, words)
            })
            .collect()
    }

    #[test]
    fn an_empty_transcript_plans_nothing() {
        assert_eq!(plan_windows(&[], &ChunkConfig::default()), []);
    }

    #[test]
    fn a_short_call_is_a_single_pass() {
        let segs = conversation(10, 12, 4_000);
        let windows = plan_windows(&segs, &ChunkConfig::default());
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start_index, 0);
        assert_eq!(windows[0].end_index, 10);
        assert_eq!(windows[0].slice(&segs).len(), 10);
    }

    #[test]
    fn a_long_call_is_split_into_map_windows() {
        // ~600 turns of 20 words ≈ 20k tokens, far past a 4k context.
        let segs = conversation(600, 20, 4_000);
        let cfg = ChunkConfig::default();
        let windows = plan_windows(&segs, &cfg);

        assert!(windows.len() > 5, "got {} windows", windows.len());
        for w in &windows {
            assert!(w.start_index < w.end_index, "empty window {w:?}");
        }
    }

    #[test]
    fn windows_cover_every_segment() {
        let segs = conversation(300, 20, 4_000);
        let windows = plan_windows(&segs, &ChunkConfig::default());

        let mut covered = vec![false; segs.len()];
        for w in &windows {
            for i in w.start_index..w.end_index {
                if let Some(c) = covered.get_mut(i as usize) {
                    *c = true;
                }
            }
        }
        assert!(covered.iter().all(|c| *c), "a segment was dropped");
    }

    #[test]
    fn consecutive_windows_overlap() {
        let segs = conversation(300, 20, 4_000);
        let cfg = ChunkConfig::default();
        let windows = plan_windows(&segs, &cfg);
        assert!(windows.len() >= 2);
        for pair in windows.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(
                b.start_index < a.end_index,
                "no overlap between {a:?} and {b:?}"
            );
            assert!(b.start_index > a.start_index, "no forward progress");
        }
    }

    #[test]
    fn windows_respect_the_token_budget() {
        let segs = conversation(400, 20, 4_000);
        let cfg = ChunkConfig::default();
        let windows = plan_windows(&segs, &cfg);
        assert!(!windows.is_empty(), "nothing to check");
        let budget = cfg.effective_window_tokens();
        for w in &windows {
            assert!(
                w.est_tokens as usize <= budget + segment_tokens(&segs[0]),
                "window {} tokens over budget {budget}",
                w.est_tokens
            );
        }
    }

    #[test]
    fn splits_prefer_a_long_silence() {
        // Uniform speaker (so speaker change never fires) with one 10 s gap
        // placed just past the seek threshold.
        let mut segs: Vec<TranscriptSegment> = Vec::new();
        let mut t = 0_u32;
        for i in 0..200 {
            // A 10 s silence every 40 turns.
            if i % 40 == 0 && i > 0 {
                t += 10_000;
            }
            segs.push(seg(Speaker::Remote, t, t + 1_500, 20));
            t += 2_000;
        }
        let windows = plan_windows(&segs, &ChunkConfig::default());
        assert!(windows.len() >= 2);

        // At least one cut lands on a gap boundary rather than mid-flow.
        let cut_at_gap = windows.iter().skip(1).any(|w| {
            let i = w.end_index as usize;
            segs.get(i)
                .zip(segs.get(i.wrapping_sub(1)))
                .is_some_and(|(next, prev)| next.start_ms.saturating_sub(prev.end_ms) >= 3_000)
        });
        assert!(cut_at_gap, "no window ended at a silence");
    }

    #[test]
    fn one_oversized_segment_becomes_its_own_window_without_looping() {
        let mut segs = conversation(4, 10, 4_000);
        segs.insert(2, seg(Speaker::Local, 8_000, 40_000, 20_000));
        let cfg = ChunkConfig::default();
        let windows = plan_windows(&segs, &cfg);

        assert_ne!(windows, []);
        assert!(windows.len() < 20, "suspicious window count");
        let giant = windows
            .iter()
            .find(|w| w.start_index <= 2 && w.end_index > 2)
            .expect("the big segment is covered");
        assert!(giant.est_tokens as usize > cfg.effective_window_tokens());
    }

    #[test]
    fn a_nonsense_config_still_produces_a_usable_plan() {
        let segs = conversation(50, 20, 4_000);
        let cfg = ChunkConfig {
            context_tokens: 0,
            reserved_tokens: 99_999,
            window_tokens: 0,
            overlap_tokens: 99_999,
            gap_split_ms: 0,
        };
        assert!(cfg.single_pass_budget() > 0);
        assert!(cfg.effective_window_tokens() >= 128);
        let windows = plan_windows(&segs, &cfg);
        assert_ne!(windows, []);
        assert!(windows
            .windows(2)
            .all(|p| p[1].start_index > p[0].start_index));
    }

    #[test]
    fn token_estimates_scale_with_words() {
        assert_eq!(estimate_tokens(""), 0);
        assert!(estimate_tokens("one two three four") >= 4);
        let s = seg(Speaker::Local, 0, 1_000, 10);
        assert_eq!(segment_tokens(&s), estimate_tokens(&s.text) + 8);
        assert_eq!(
            transcript_tokens(&[s.clone(), s.clone()]),
            segment_tokens(&s) * 2
        );
    }

    #[test]
    fn head_and_tail_pick_the_greeting_and_the_close() {
        let segs = conversation(60, 10, 5_000); // 0 .. ~297 s
        let (head, tail) = head_and_tail(&segs, 60);
        assert_ne!(head, []);
        assert_ne!(tail, []);
        assert_eq!(head.first().map(|s| s.start_ms), Some(0));
        assert_eq!(tail.last().map(|s| s.end_ms), segs.last().map(|s| s.end_ms));
        // They must not overlap.
        assert!(head.len() + tail.len() <= segs.len());
    }

    #[test]
    fn head_and_tail_of_a_short_call_do_not_double_count() {
        let segs = conversation(3, 10, 2_000);
        let (head, tail) = head_and_tail(&segs, 600);
        assert_eq!(head.len(), 3);
        assert_eq!(tail, []);
    }

    #[test]
    fn head_and_tail_of_nothing_is_nothing() {
        let (h, t) = head_and_tail(&[], 60);
        assert!(h.is_empty() && t.is_empty());
    }
}

//! Speaker-attributed segmentation of the two call legs.
//!
//! The two legs are never summed (design spec §3.2), so attribution is a
//! bookkeeping problem rather than a diarization problem: run VAD per leg,
//! label the spans, merge the timelines. The one real complication is echo —
//! the far end's voice bleeding into the local microphone — handled by the
//! cross-leg energy gate the spec asks for.

use crate::audio::vad::{detect_speech, rms, SpeechSpan, VadConfig};
use crate::types::Speaker;
use crate::util::ms_to_samples;

/// Tunables for [`segment_call`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct SegmentationConfig {
    pub vad: VadConfig,
    /// A local span is treated as echo when the remote leg is at least this
    /// many times louder across the overlapping window.
    pub echo_ratio: f32,
    /// …and at least this fraction of the local span is overlapped by remote
    /// speech. Both conditions must hold, so genuine interruption survives.
    pub echo_min_overlap: f32,
    /// Hard ceiling on a segment handed to the STT engine. Kept under
    /// Whisper's 30 s receptive field; also bounds peak memory for Parakeet.
    pub max_segment_ms: u32,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            vad: VadConfig::default(),
            echo_ratio: 4.0,
            echo_min_overlap: 0.7,
            max_segment_ms: 28_000,
        }
    }
}

/// A timed, attributed slice of one leg, ready to hand to the STT engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct AudioSegment {
    pub speaker: Speaker,
    pub start_ms: u32,
    pub end_ms: u32,
}

impl AudioSegment {
    /// Length of the segment.
    #[must_use]
    pub const fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Borrow the `[start_ms, end_ms)` window of `samples`, clamped to the buffer.
///
/// Returns an empty slice rather than panicking for out-of-range windows —
/// this is on the FFI-reachable path.
#[must_use]
pub fn slice_ms(samples: &[f32], rate_hz: u32, start_ms: u32, end_ms: u32) -> &[f32] {
    if rate_hz == 0 || end_ms <= start_ms {
        return &[];
    }
    let start = ms_to_samples(u64::from(start_ms), rate_hz).min(samples.len());
    let end = ms_to_samples(u64::from(end_ms), rate_hz).min(samples.len());
    samples.get(start..end).unwrap_or(&[])
}

/// Produce the merged, speaker-attributed segment timeline for a call.
///
/// `local` and `remote` must both already be at `rate_hz` (16 kHz in
/// practice) and share a `t = 0` origin — which they do, because the RTP
/// tap pushes both legs on the same 20 ms grid, substituting silence on
/// underrun (design spec §3.1).
#[must_use]
pub fn segment_call(
    local: &[f32],
    remote: &[f32],
    rate_hz: u32,
    cfg: &SegmentationConfig,
) -> Vec<AudioSegment> {
    let remote_spans = detect_speech(remote, rate_hz, &cfg.vad);
    let local_spans = detect_speech(local, rate_hz, &cfg.vad);

    let kept_local: Vec<SpeechSpan> = local_spans
        .into_iter()
        .filter(|span| !is_echo(*span, local, remote, rate_hz, &remote_spans, cfg))
        .collect();

    let mut out: Vec<AudioSegment> = Vec::new();
    push_split(&mut out, Speaker::Local, &kept_local, cfg.max_segment_ms);
    push_split(&mut out, Speaker::Remote, &remote_spans, cfg.max_segment_ms);

    // Stable order: by start, then Local before Remote so overlapping turns
    // render deterministically.
    out.sort_by_key(|s| {
        (
            s.start_ms,
            match s.speaker {
                Speaker::Local => 0_u8,
                Speaker::Remote => 1,
            },
            s.end_ms,
        )
    });
    out
}

fn push_split(out: &mut Vec<AudioSegment>, speaker: Speaker, spans: &[SpeechSpan], max_ms: u32) {
    let max_ms = max_ms.max(1_000);
    for span in spans {
        let mut start = span.start_ms;
        while start < span.end_ms {
            let end = start.saturating_add(max_ms).min(span.end_ms);
            out.push(AudioSegment {
                speaker,
                start_ms: start,
                end_ms: end,
            });
            if end == span.end_ms {
                break;
            }
            start = end;
        }
    }
}

/// Cross-leg energy gate: is this local span just the far end leaking into
/// the microphone?
fn is_echo(
    span: SpeechSpan,
    local: &[f32],
    remote: &[f32],
    rate_hz: u32,
    remote_spans: &[SpeechSpan],
    cfg: &SegmentationConfig,
) -> bool {
    let dur = span.duration_ms();
    if dur == 0 {
        return false;
    }
    let overlap: u32 = remote_spans.iter().map(|r| span.overlap_ms(r)).sum();
    #[allow(clippy::cast_precision_loss)]
    let frac = f32::from(u16::try_from(overlap.min(u32::from(u16::MAX))).unwrap_or(u16::MAX))
        / f32::from(u16::try_from(dur.min(u32::from(u16::MAX))).unwrap_or(u16::MAX));
    if frac < cfg.echo_min_overlap.clamp(0.0, 1.0) {
        return false;
    }

    let local_rms = rms(slice_ms(local, rate_hz, span.start_ms, span.end_ms));
    let remote_rms = rms(slice_ms(remote, rate_hz, span.start_ms, span.end_ms));
    if local_rms <= f32::EPSILON {
        return false;
    }
    remote_rms >= local_rms * cfg.echo_ratio.max(1.0)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn samples(ms: u32) -> usize {
        ms_to_samples(u64::from(ms), RATE)
    }

    fn tone(ms: u32, amp: f32, hz: f32) -> Vec<f32> {
        (0..samples(ms))
            .map(|i| {
                let t = i as f32 / RATE as f32;
                amp * (2.0 * std::f32::consts::PI * hz * t).sin()
            })
            .collect()
    }

    fn silence(ms: u32) -> Vec<f32> {
        vec![0.0; samples(ms)]
    }

    /// Build a leg by concatenating `(duration_ms, amplitude)` blocks.
    fn leg(blocks: &[(u32, f32)]) -> Vec<f32> {
        let mut v = Vec::new();
        for (ms, amp) in blocks {
            if *amp <= 0.0 {
                v.extend(silence(*ms));
            } else {
                v.extend(tone(*ms, *amp, 220.0));
            }
        }
        v
    }

    #[test]
    fn a_one_sided_conversation_is_attributed_to_that_leg() {
        let local = leg(&[(500, 0.0), (900, 0.5), (500, 0.0)]);
        let remote = silence(1_900);
        let segs = segment_call(&local, &remote, RATE, &SegmentationConfig::default());
        assert_eq!(segs.len(), 1, "{segs:?}");
        assert_eq!(segs[0].speaker, Speaker::Local);

        let segs = segment_call(&remote, &local, RATE, &SegmentationConfig::default());
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker, Speaker::Remote);
    }

    #[test]
    fn alternating_turns_interleave_in_time_order() {
        // Them 0.0-0.8, Me 1.5-2.3, Them 3.0-3.8
        let remote = leg(&[(800, 0.5), (700, 0.0), (800, 0.0), (700, 0.0), (800, 0.5)]);
        let local = leg(&[(800, 0.0), (700, 0.0), (800, 0.5), (700, 0.0), (800, 0.0)]);

        let segs = segment_call(&local, &remote, RATE, &SegmentationConfig::default());
        assert_eq!(segs.len(), 3, "{segs:?}");
        assert_eq!(
            segs.iter().map(|s| s.speaker).collect::<Vec<_>>(),
            vec![Speaker::Remote, Speaker::Local, Speaker::Remote]
        );
        assert!(segs.windows(2).all(|w| w[0].start_ms <= w[1].start_ms));
    }

    #[test]
    fn echo_of_the_far_end_into_the_mic_is_suppressed() {
        // Remote talks loudly; the same waveform leaks into local at 1/10th.
        let remote = leg(&[(300, 0.0), (1_200, 0.6), (300, 0.0)]);
        let local: Vec<f32> = remote.iter().map(|s| s * 0.1).collect();

        let cfg = SegmentationConfig::default();
        // Sanity: the leaked signal is loud enough that VAD alone would take it.
        assert_ne!(detect_speech(&local, RATE, &cfg.vad), []);

        let segs = segment_call(&local, &remote, RATE, &cfg);
        // Both halves matter: the echo must go *and* the real speech must
        // stay. Asserting only `all(Remote)` would pass on an empty result.
        assert!(
            segs.iter().any(|s| s.speaker == Speaker::Remote),
            "the far end was suppressed along with its echo: {segs:?}"
        );
        assert!(
            segs.iter().all(|s| s.speaker == Speaker::Remote),
            "echo survived: {segs:?}"
        );
    }

    #[test]
    fn genuine_simultaneous_speech_survives_the_echo_gate() {
        // Both talking at comparable level: this is interruption, not echo.
        let remote = leg(&[(300, 0.0), (1_200, 0.6), (300, 0.0)]);
        let local = leg(&[(300, 0.0), (1_200, 0.45), (300, 0.0)]);

        let segs = segment_call(&local, &remote, RATE, &SegmentationConfig::default());
        assert!(segs.iter().any(|s| s.speaker == Speaker::Local), "{segs:?}");
        assert!(segs.iter().any(|s| s.speaker == Speaker::Remote));
    }

    #[test]
    fn quiet_local_speech_with_a_silent_far_end_is_not_gated() {
        // Low overlap => never echo, regardless of level.
        let local = leg(&[(200, 0.0), (1_000, 0.08), (200, 0.0)]);
        let remote = silence(1_400);
        let segs = segment_call(&local, &remote, RATE, &SegmentationConfig::default());
        assert!(segs.iter().any(|s| s.speaker == Speaker::Local), "{segs:?}");
    }

    #[test]
    fn long_spans_are_split_below_the_model_window() {
        let local = tone(70_000, 0.5, 220.0);
        let remote = silence(70_000);
        let cfg = SegmentationConfig {
            max_segment_ms: 28_000,
            ..SegmentationConfig::default()
        };
        let segs = segment_call(&local, &remote, RATE, &cfg);
        assert!(segs.len() >= 3, "{}", segs.len());
        assert!(segs.iter().all(|s| s.duration_ms() <= 28_000));
        // Contiguous, no gaps or overlaps introduced by the split.
        assert!(segs.windows(2).all(|w| w[0].end_ms == w[1].start_ms));
    }

    #[test]
    fn slice_ms_clamps_instead_of_panicking() {
        let buf = vec![1.0_f32; samples(100)];
        assert_eq!(slice_ms(&buf, RATE, 0, 50).len(), samples(50));
        assert_eq!(slice_ms(&buf, RATE, 90, 5_000).len(), samples(10));
        assert_eq!(slice_ms(&buf, RATE, 500, 600), [] as [f32; 0]);
        assert_eq!(slice_ms(&buf, RATE, 50, 50), [] as [f32; 0]);
        assert_eq!(slice_ms(&buf, 0, 0, 50), [] as [f32; 0]);
        assert_eq!(slice_ms(&[], RATE, 0, 50), [] as [f32; 0]);
    }

    #[test]
    fn two_silent_legs_produce_nothing() {
        let segs = segment_call(
            &silence(5_000),
            &silence(5_000),
            RATE,
            &SegmentationConfig::default(),
        );
        assert_eq!(segs, []);
    }
}

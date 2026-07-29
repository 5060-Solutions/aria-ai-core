//! Energy-based voice activity detection.
//!
//! Purpose is not perfect speech detection — it is (a) cutting the long
//! non-vocal stretches that phone calls are full of, because those are the
//! documented trigger for ASR hallucination (model research §6.3), and (b)
//! giving the segmenter turn boundaries to attribute speakers with.
//!
//! Deliberately deterministic and dependency-free so it is unit-testable with
//! synthetic audio and identical on every platform. Silero VAD is a Phase-6
//! upgrade that slots in behind the same [`SpeechSpan`] output.

use crate::util::{ms_to_samples, samples_to_ms};

/// Tunables for [`detect_speech`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct VadConfig {
    /// Analysis frame length. 20 ms matches the RTP grid.
    pub frame_ms: u32,
    /// Absolute RMS floor. Nothing quieter is ever speech, whatever the
    /// noise estimate says. Guards against amplifying pure digital silence,
    /// and is what makes steady low-level line noise a non-event.
    pub abs_floor_rms: f32,
    /// Speech must exceed the estimated noise floor by this factor.
    pub snr_ratio: f32,
    /// Quantile of frame energies used as the noise-floor estimate, in `[0,1]`.
    pub noise_quantile: f32,
    /// Ceiling on the adaptive threshold, as a fraction of the loudest frame.
    ///
    /// Without this, a leg that is *mostly* speech has its own speech sitting
    /// at the noise quantile, the adaptive threshold climbs above the signal,
    /// and the whole leg reads as silent. Capping the threshold at a fraction
    /// of the peak makes a continuously-talking leg work.
    pub peak_ratio: f32,
    /// Trailing padding added to each detected run, so word endings survive.
    pub hangover_ms: u32,
    /// Runs shorter than this are discarded as clicks/comfort noise.
    pub min_speech_ms: u32,
    /// Runs separated by less than this are merged into one.
    pub merge_gap_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_ms: 20,
            abs_floor_rms: 0.02,
            snr_ratio: 3.0,
            noise_quantile: 0.2,
            peak_ratio: 0.25,
            hangover_ms: 200,
            min_speech_ms: 150,
            merge_gap_ms: 300,
        }
    }
}

/// A contiguous run of detected speech on one leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct SpeechSpan {
    pub start_ms: u32,
    pub end_ms: u32,
}

impl SpeechSpan {
    /// Length of the span.
    #[must_use]
    pub const fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Overlap in milliseconds with `other`, zero if disjoint.
    #[must_use]
    pub const fn overlap_ms(&self, other: &Self) -> u32 {
        let start = if self.start_ms > other.start_ms {
            self.start_ms
        } else {
            other.start_ms
        };
        let end = if self.end_ms < other.end_ms {
            self.end_ms
        } else {
            other.end_ms
        };
        end.saturating_sub(start)
    }
}

/// Per-frame RMS energies for `samples`, one entry per `cfg.frame_ms`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn frame_energies(samples: &[f32], rate_hz: u32, frame_ms: u32) -> Vec<f32> {
    let frame = ms_to_samples(u64::from(frame_ms.max(1)), rate_hz).max(1);
    samples
        .chunks(frame)
        .map(|c| {
            if c.is_empty() {
                0.0
            } else {
                (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt()
            }
        })
        .collect()
}

/// RMS of a slice. Zero for an empty slice — never NaN.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Locate runs of speech in one leg of 16 kHz mono audio.
///
/// Returns spans sorted by start time, non-overlapping, clamped to the buffer.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn detect_speech(samples: &[f32], rate_hz: u32, cfg: &VadConfig) -> Vec<SpeechSpan> {
    if samples.is_empty() || rate_hz == 0 {
        return Vec::new();
    }
    let frame_ms = cfg.frame_ms.max(1);
    let energies = frame_energies(samples, rate_hz, frame_ms);
    if energies.is_empty() {
        return Vec::new();
    }

    let threshold = speech_threshold(&energies, cfg);
    let total_ms = samples_to_ms(samples.len(), rate_hz);

    // 1. Raw runs of above-threshold frames.
    let mut runs: Vec<SpeechSpan> = Vec::new();
    let mut open: Option<u32> = None;
    for (i, e) in energies.iter().enumerate() {
        let start_ms = (i as u64 * u64::from(frame_ms)) as u32;
        if *e >= threshold {
            if open.is_none() {
                open = Some(start_ms);
            }
        } else if let Some(s) = open.take() {
            runs.push(SpeechSpan {
                start_ms: s,
                end_ms: start_ms,
            });
        }
    }
    if let Some(s) = open {
        runs.push(SpeechSpan {
            start_ms: s,
            end_ms: u32::try_from(total_ms).unwrap_or(u32::MAX),
        });
    }

    // 2. Hangover.
    for r in &mut runs {
        r.end_ms = r
            .end_ms
            .saturating_add(cfg.hangover_ms)
            .min(u32::try_from(total_ms).unwrap_or(u32::MAX));
    }

    // 3. Merge close runs.
    let mut merged: Vec<SpeechSpan> = Vec::with_capacity(runs.len());
    for r in runs {
        match merged.last_mut() {
            Some(prev) if r.start_ms <= prev.end_ms.saturating_add(cfg.merge_gap_ms) => {
                prev.end_ms = prev.end_ms.max(r.end_ms);
            }
            _ => merged.push(r),
        }
    }

    // 4. Drop runs that are too short to be speech.
    merged.retain(|r| r.duration_ms() >= cfg.min_speech_ms);
    merged
}

/// Sum of span durations.
#[must_use]
pub fn total_speech_ms(spans: &[SpeechSpan]) -> u64 {
    spans.iter().map(|s| u64::from(s.duration_ms())).sum()
}

/// Threshold = `max(absolute floor, min(noise × SNR, peak × peak_ratio))`.
///
/// The `min` is what stops a continuously-talking leg from suppressing
/// itself; the `max` is what stops steady low-level line noise from reading
/// as speech. An energy VAD cannot separate loud steady noise from speech —
/// that is what Silero VAD is for, in a later phase.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn speech_threshold(energies: &[f32], cfg: &VadConfig) -> f32 {
    let mut sorted: Vec<f32> = energies.to_vec();
    sorted.sort_by(f32::total_cmp);
    let q = cfg.noise_quantile.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f32 * q) as usize;
    let noise = sorted.get(idx).copied().unwrap_or(0.0);
    let peak = sorted.last().copied().unwrap_or(0.0);

    let adaptive = (noise * cfg.snr_ratio.max(1.0)).min(peak * cfg.peak_ratio.clamp(0.01, 1.0));
    adaptive.max(cfg.abs_floor_rms.max(0.0))
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn silence(ms: u32) -> Vec<f32> {
        vec![0.0; crate::util::ms_to_samples(u64::from(ms), RATE)]
    }

    fn noise(ms: u32, amp: f32) -> Vec<f32> {
        // Deterministic pseudo-noise: alternating +/- amp is worst case for RMS.
        (0..crate::util::ms_to_samples(u64::from(ms), RATE))
            .map(|i| if i % 2 == 0 { amp } else { -amp })
            .collect()
    }

    fn speech(ms: u32, amp: f32) -> Vec<f32> {
        (0..crate::util::ms_to_samples(u64::from(ms), RATE))
            .map(|i| {
                let t = i as f32 / RATE as f32;
                amp * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn pure_silence_has_no_speech() {
        let spans = detect_speech(&silence(3_000), RATE, &VadConfig::default());
        assert!(spans.is_empty(), "{spans:?}");
    }

    #[test]
    fn steady_line_noise_below_the_absolute_floor_is_not_speech() {
        let spans = detect_speech(&noise(3_000, 0.01), RATE, &VadConfig::default());
        assert!(spans.is_empty(), "{spans:?}");
    }

    #[test]
    fn the_adaptive_threshold_lifts_speech_out_of_audible_background_noise() {
        // Background well above the absolute floor, with one burst on top.
        let mut audio = noise(1_000, 0.06);
        audio.extend(speech(700, 0.4));
        audio.extend(noise(1_000, 0.06));

        let spans = detect_speech(&audio, RATE, &VadConfig::default());
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert!(spans[0].start_ms.abs_diff(1_000) <= 60, "{:?}", spans[0]);
    }

    #[test]
    fn a_leg_that_never_stops_talking_is_all_speech() {
        // The failure mode the peak ratio exists to prevent: with no silence
        // anywhere, the noise quantile lands on speech itself.
        let spans = detect_speech(&speech(5_000, 0.5), RATE, &VadConfig::default());
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert!(spans[0].duration_ms() >= 4_900, "{:?}", spans[0]);
    }

    #[test]
    fn a_single_burst_is_found_at_the_right_time() {
        let mut audio = silence(1_000);
        audio.extend(speech(800, 0.4));
        audio.extend(silence(1_000));

        let spans = detect_speech(&audio, RATE, &VadConfig::default());
        assert_eq!(spans.len(), 1, "{spans:?}");
        let s = spans[0];
        assert!(s.start_ms.abs_diff(1_000) <= 40, "start {}", s.start_ms);
        // End includes the 200 ms hangover.
        assert!(s.end_ms.abs_diff(2_000) <= 60, "end {}", s.end_ms);
    }

    #[test]
    fn a_long_gap_splits_but_a_short_gap_merges() {
        let mut audio = speech(600, 0.4);
        audio.extend(silence(2_000)); // way beyond merge_gap
        audio.extend(speech(600, 0.4));
        let spans = detect_speech(&audio, RATE, &VadConfig::default());
        assert_eq!(spans.len(), 2, "{spans:?}");

        let mut audio = speech(600, 0.4);
        audio.extend(silence(120)); // inside hangover + merge_gap
        audio.extend(speech(600, 0.4));
        let spans = detect_speech(&audio, RATE, &VadConfig::default());
        assert_eq!(spans.len(), 1, "{spans:?}");
    }

    #[test]
    fn a_click_shorter_than_min_speech_is_discarded() {
        let mut audio = silence(1_000);
        audio.extend(speech(20, 0.9)); // one frame
        audio.extend(silence(1_000));
        let cfg = VadConfig {
            hangover_ms: 0,
            ..VadConfig::default()
        };
        assert!(detect_speech(&audio, RATE, &cfg).is_empty());
    }

    #[test]
    fn empty_or_degenerate_input_is_handled() {
        assert!(detect_speech(&[], RATE, &VadConfig::default()).is_empty());
        assert!(detect_speech(&silence(100), 0, &VadConfig::default()).is_empty());
        let weird = VadConfig {
            frame_ms: 0,
            noise_quantile: 5.0,
            snr_ratio: -1.0,
            peak_ratio: -3.0,
            ..VadConfig::default()
        };
        // Must not panic on nonsense config.
        let _ = detect_speech(&speech(500, 0.5), RATE, &weird);
    }

    #[test]
    fn overlap_maths() {
        let a = SpeechSpan {
            start_ms: 100,
            end_ms: 500,
        };
        let b = SpeechSpan {
            start_ms: 400,
            end_ms: 900,
        };
        let c = SpeechSpan {
            start_ms: 900,
            end_ms: 1_000,
        };
        assert_eq!(a.overlap_ms(&b), 100);
        assert_eq!(b.overlap_ms(&a), 100);
        assert_eq!(a.overlap_ms(&c), 0);
        assert_eq!(a.duration_ms(), 400);
    }

    #[test]
    fn rms_of_empty_is_zero_not_nan() {
        assert!((rms(&[]) - 0.0).abs() < f32::EPSILON);
        assert!(rms(&[0.5, -0.5]).is_finite());
    }

    #[test]
    fn total_speech_sums_spans() {
        let spans = vec![
            SpeechSpan {
                start_ms: 0,
                end_ms: 1_000,
            },
            SpeechSpan {
                start_ms: 2_000,
                end_ms: 2_500,
            },
        ];
        assert_eq!(total_speech_ms(&spans), 1_500);
    }
}

//! Band-limited rational resampling to the 16 kHz mono `f32` that every
//! Whisper-class / Parakeet-class front end expects.
//!
//! Why not linear interpolation: `rtp_engine::resample::StreamResampler` is
//! linear, which is fine for playback and wrong ahead of an ASR front end —
//! it is not band-limited, so decimating 48 kHz Opus with it folds everything
//! above 8 kHz back into the speech band. The model research is explicit that
//! plain band-limited sinc is the *best measured* 8→16 kHz choice (better than
//! every neural bandwidth-extension method tested), so that is what this is:
//! a windowed-sinc polyphase FIR, Kaiser window, exact rational ratio.
//!
//! The two ratios that matter in practice are integers:
//!   * PCMU/PCMA/G.729 8 kHz  → 16 kHz  (L=2, M=1)
//!   * Opus            48 kHz → 16 kHz  (L=1, M=3)
//!
//! The implementation handles any rate pair via `gcd`, so a host that hands us
//! 44.1 kHz still works.
//!
//! Group delay is compensated internally: output sample `n` corresponds to
//! input time `n * M / L`, so the two call legs stay on the same timeline and
//! `merge`-by-timestamp speaker attribution remains correct.

use crate::error::{AiError, Result};

/// The rate every downstream model consumes.
pub const TARGET_RATE_HZ: u32 = 16_000;

/// Taps per polyphase branch, each side of centre. 16 → 33-tap branches,
/// roughly 85 dB stopband with the Kaiser beta below. Cheap enough to run on
/// a phone in the audio drain thread.
const HALF_TAPS: usize = 16;

/// Kaiser beta. 8.6 ≈ 85 dB stopband attenuation.
const KAISER_BETA: f32 = 8.6;

/// Pull the cutoff slightly below Nyquist of the lower rate to keep the
/// transition band off the very top of the passband. At 8→16 kHz this puts
/// the corner near 3.8 kHz, which is above the 3.4 kHz G.711 passband, so no
/// real telephony content is lost.
const CUTOFF_SCALE: f32 = 0.95;

/// Highest input rate we will accept from a platform audio bridge.
const MAX_RATE_HZ: u32 = 192_000;

/// Streaming, band-limited, rational-ratio resampler.
///
/// Feed it `i16` frames as they arrive from the RTP thread; it emits 16 kHz
/// `f32` in `[-1.0, 1.0]`. State is carried across calls, so chunked input
/// produces bit-identical output to a single-shot call.
#[derive(Debug, Clone)]
pub struct SincResampler {
    in_rate: u32,
    out_rate: u32,
    /// Interpolation factor.
    l: usize,
    /// Decimation factor.
    m: usize,
    /// Prototype low-pass, designed at `l * in_rate`, scaled for unity DC gain.
    taps: Vec<f32>,
    /// Index of the filter's centre tap; the group-delay compensation.
    center: usize,
    /// Rolling input window. `history[0]` is absolute input index `hist_start`.
    history: Vec<f32>,
    hist_start: i64,
    /// Absolute index of the next output sample to produce.
    next_out: u64,
    /// `in_rate == out_rate`: skip the filter entirely.
    passthrough: bool,
}

impl SincResampler {
    /// Build a resampler from `in_rate` to `out_rate`.
    ///
    /// # Errors
    /// [`AiError::UnsupportedSampleRate`] if either rate is zero or above
    /// [`MAX_RATE_HZ`].
    pub fn new(in_rate: u32, out_rate: u32) -> Result<Self> {
        if in_rate == 0 || in_rate > MAX_RATE_HZ {
            return Err(AiError::UnsupportedSampleRate { rate: in_rate });
        }
        if out_rate == 0 || out_rate > MAX_RATE_HZ {
            return Err(AiError::UnsupportedSampleRate { rate: out_rate });
        }

        if in_rate == out_rate {
            return Ok(Self {
                in_rate,
                out_rate,
                l: 1,
                m: 1,
                taps: Vec::new(),
                center: 0,
                history: Vec::new(),
                hist_start: 0,
                next_out: 0,
                passthrough: true,
            });
        }

        let g = gcd(in_rate, out_rate);
        let l = (out_rate / g) as usize;
        let m = (in_rate / g) as usize;
        let center = HALF_TAPS * l;
        let taps = design_lowpass(l, m, center);

        Ok(Self {
            in_rate,
            out_rate,
            l,
            m,
            taps,
            center,
            history: Vec::new(),
            hist_start: 0,
            next_out: 0,
            passthrough: false,
        })
    }

    /// Build a resampler from `in_rate` to [`TARGET_RATE_HZ`].
    ///
    /// # Errors
    /// As [`SincResampler::new`].
    pub fn to_target(in_rate: u32) -> Result<Self> {
        Self::new(in_rate, TARGET_RATE_HZ)
    }

    /// Input rate this resampler was built for.
    #[must_use]
    pub const fn input_rate(&self) -> u32 {
        self.in_rate
    }

    /// Output rate this resampler produces.
    #[must_use]
    pub const fn output_rate(&self) -> u32 {
        self.out_rate
    }

    /// Number of *input* samples of lookahead held before output can be emitted.
    /// Equivalent to the compensated group delay.
    #[must_use]
    pub const fn lookahead_samples(&self) -> usize {
        if self.passthrough {
            0
        } else {
            HALF_TAPS
        }
    }

    /// Push `i16` input and append whatever output is now available.
    ///
    /// Never allocates in the hot path beyond growing `out`, never blocks,
    /// never panics.
    pub fn process_i16(&mut self, input: &[i16], out: &mut Vec<f32>) {
        if self.passthrough {
            out.reserve(input.len());
            out.extend(input.iter().map(|s| f32::from(*s) / 32768.0));
            return;
        }
        self.history
            .extend(input.iter().map(|s| f32::from(*s) / 32768.0));
        self.produce(out);
        self.trim();
    }

    /// Push `f32` input and append whatever output is now available.
    pub fn process_f32(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.passthrough {
            out.extend_from_slice(input);
            return;
        }
        self.history.extend_from_slice(input);
        self.produce(out);
        self.trim();
    }

    /// Flush the filter tail with zero-padding, then reset the stream position.
    ///
    /// Call once when a call leg ends so the last ~1 ms of speech is emitted.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if self.passthrough {
            return;
        }
        let pad = vec![0.0_f32; HALF_TAPS + 1];
        self.history.extend_from_slice(&pad);
        self.produce(out);
        self.reset();
    }

    /// Drop all state and restart at output index 0.
    pub fn reset(&mut self) {
        self.history.clear();
        self.hist_start = 0;
        self.next_out = 0;
    }

    /// Convenience: resample a whole buffer in one shot, including the flush.
    ///
    /// # Errors
    /// As [`SincResampler::new`].
    pub fn resample_all(in_rate: u32, out_rate: u32, input: &[i16]) -> Result<Vec<f32>> {
        let mut r = Self::new(in_rate, out_rate)?;
        let mut out = Vec::with_capacity(input.len() * 2);
        r.process_i16(input, &mut out);
        r.flush(&mut out);
        Ok(out)
    }

    fn sample_at(&self, idx: i64) -> f32 {
        if idx < self.hist_start {
            // Pre-roll before the stream started: implicit silence.
            return 0.0;
        }
        let Ok(off) = usize::try_from(idx - self.hist_start) else {
            return 0.0;
        };
        self.history.get(off).copied().unwrap_or(0.0)
    }

    fn produce(&mut self, out: &mut Vec<f32>) {
        let l = self.l as u64;
        let m = self.m as u64;
        let center = self.center as u64;
        let Ok(hist_len) = i64::try_from(self.history.len()) else {
            return;
        };
        let last_idx = self.hist_start + hist_len - 1;

        loop {
            let q = self.next_out * m + center;
            let Ok(base) = i64::try_from(q / l) else {
                return;
            };
            if base > last_idx {
                break;
            }
            let phase = (q % l) as usize;

            let mut acc = 0.0_f32;
            let mut tap = phase;
            let mut back: i64 = 0;
            while tap < self.taps.len() {
                acc = self.taps[tap].mul_add(self.sample_at(base - back), acc);
                tap += self.l;
                back += 1;
            }
            out.push(acc);
            self.next_out += 1;
        }
    }

    fn trim(&mut self) {
        let l = self.l as u64;
        let m = self.m as u64;
        let q = self.next_out * m + self.center as u64;
        let Ok(next_base) = i64::try_from(q / l) else {
            return;
        };
        let span = ((self.taps.len() - 1) / self.l) as i64;
        let keep_from = next_base - span;
        if keep_from <= self.hist_start {
            return;
        }
        let Ok(drop) = usize::try_from(keep_from - self.hist_start) else {
            return;
        };
        let drop = drop.min(self.history.len());
        self.history.drain(..drop);
        self.hist_start += drop as i64;
    }
}

const fn gcd(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Windowed-sinc prototype, normalised so the overall DC gain is 1.0.
#[allow(clippy::cast_precision_loss)] // tap indices are small; f32 is exact here
fn design_lowpass(up: usize, down: usize, center: usize) -> Vec<f32> {
    let n_taps = 2 * center + 1;
    // Normalised to the interpolated rate `up * in_rate`.
    let fc = CUTOFF_SCALE * 0.5 / (up.max(down) as f32);

    let mut taps = Vec::with_capacity(n_taps);
    let denom = bessel_i0(KAISER_BETA);
    let half = center as f32;
    for idx in 0..n_taps {
        let offset = idx as f32 - half;
        let ideal = 2.0 * fc * sinc(2.0 * fc * offset);
        // Kaiser window.
        let ratio = offset / half;
        let arg = ratio.mul_add(-ratio, 1.0);
        let window = if arg <= 0.0 {
            0.0
        } else {
            bessel_i0(KAISER_BETA * arg.sqrt()) / denom
        };
        taps.push(ideal * window);
    }

    // Unity DC gain: each polyphase branch must sum to 1, so the whole
    // prototype must sum to L.
    let sum: f32 = taps.iter().sum();
    if sum.abs() > f32::EPSILON {
        let scale = up as f32 / sum;
        for t in &mut taps {
            *t *= scale;
        }
    }
    taps
}

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let pix = std::f32::consts::PI * x;
        pix.sin() / pix
    }
}

/// Modified Bessel function of the first kind, order 0. Series expansion is
/// plenty accurate for the beta range we use and avoids a maths dependency.
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0_f32;
    let mut term = 1.0_f32;
    let half_x = x / 2.0;
    for k in 1..=24_u32 {
        let k_f = k as f32;
        term *= (half_x / k_f) * (half_x / k_f);
        sum += term;
        if term < sum * 1e-9 {
            break;
        }
    }
    sum
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    fn rms(v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt()
    }

    fn tone(rate: u32, hz: f32, secs: f32, amp: f32) -> Vec<i16> {
        let n = (rate as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (amp * (2.0 * std::f32::consts::PI * hz * t).sin() * 32767.0) as i16
            })
            .collect()
    }

    #[test]
    fn rejects_impossible_rates() {
        assert!(matches!(
            SincResampler::new(0, 16_000),
            Err(AiError::UnsupportedSampleRate { rate: 0 })
        ));
        assert!(matches!(
            SincResampler::new(8_000, 0),
            Err(AiError::UnsupportedSampleRate { rate: 0 })
        ));
        assert!(SincResampler::new(1_000_000, 16_000).is_err());
    }

    #[test]
    fn same_rate_is_passthrough() {
        let input: Vec<i16> = vec![0, 16_384, -16_384, 32_767, -32_768];
        let out = SincResampler::resample_all(16_000, 16_000, &input).expect("build");
        assert_eq!(out.len(), input.len());
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[3] - 32_767.0 / 32_768.0).abs() < 1e-6);
    }

    #[test]
    fn upsample_8k_to_16k_doubles_the_sample_count() {
        // 1 second of 8 kHz -> ~1 second of 16 kHz.
        let input = tone(8_000, 440.0, 1.0, 0.5);
        let out = SincResampler::resample_all(8_000, 16_000, &input).expect("build");
        let expected = input.len() * 2;
        // Tolerance is the filter tail, a couple of samples either way.
        assert!(
            out.len().abs_diff(expected) <= 4,
            "got {}, expected ~{expected}",
            out.len()
        );
    }

    #[test]
    fn downsample_48k_to_16k_thirds_the_sample_count() {
        let input = tone(48_000, 440.0, 1.0, 0.5);
        let out = SincResampler::resample_all(48_000, 16_000, &input).expect("build");
        let expected = input.len() / 3;
        assert!(
            out.len().abs_diff(expected) <= 4,
            "got {}, expected ~{expected}",
            out.len()
        );
    }

    #[test]
    fn dc_gain_is_unity_for_both_telephony_ratios() {
        for (from, to) in [(8_000_u32, 16_000_u32), (48_000, 16_000)] {
            let input = vec![16_384_i16; from as usize]; // constant 0.5
            let out = SincResampler::resample_all(from, to, &input).expect("build");
            // Ignore the leading/trailing transition regions.
            let skip = out.len() / 10;
            let steady = &out[skip..out.len() - skip];
            let mean = steady.iter().sum::<f32>() / steady.len() as f32;
            assert!(
                (mean - 0.5).abs() < 5e-3,
                "{from}->{to}: DC gain {mean}, expected 0.5"
            );
        }
    }

    #[test]
    fn speech_band_tone_survives_8k_upsampling_with_amplitude_intact() {
        // 1 kHz sits squarely in the G.711 passband.
        let input = tone(8_000, 1_000.0, 0.5, 0.5);
        let out = SincResampler::resample_all(8_000, 16_000, &input).expect("build");
        let skip = out.len() / 10;
        let steady = &out[skip..out.len() - skip];
        let expected = 0.5 / std::f32::consts::SQRT_2; // RMS of a 0.5-amplitude sine
        assert!(
            (rms(steady) - expected).abs() < 0.02,
            "rms {} expected {expected}",
            rms(steady)
        );
    }

    #[test]
    fn decimation_rejects_content_above_the_new_nyquist() {
        // 15 kHz at 48 kHz input is far above the 8 kHz output Nyquist. A
        // band-limited filter must kill it; linear interpolation would alias
        // it down into the speech band.
        let input = tone(48_000, 15_000.0, 0.5, 0.9);
        let out = SincResampler::resample_all(48_000, 16_000, &input).expect("build");
        let skip = out.len() / 10;
        let steady = &out[skip..out.len() - skip];
        assert!(
            rms(steady) < 0.01,
            "aliased energy leaked through: rms {}",
            rms(steady)
        );

        // Sanity: the same amplitude at 1 kHz does pass.
        let pass = tone(48_000, 1_000.0, 0.5, 0.9);
        let out2 = SincResampler::resample_all(48_000, 16_000, &pass).expect("build");
        let steady2 = &out2[skip..out2.len() - skip];
        assert!(rms(steady2) > 0.5, "passband was attenuated: {}", rms(steady2));
    }

    #[test]
    fn chunked_input_matches_single_shot_exactly() {
        let input = tone(8_000, 700.0, 0.4, 0.4);

        let single = SincResampler::resample_all(8_000, 16_000, &input).expect("build");

        let mut r = SincResampler::new(8_000, 16_000).expect("build");
        let mut chunked = Vec::new();
        // 20 ms RTP frames, exactly how the media session delivers them.
        for frame in input.chunks(160) {
            r.process_i16(frame, &mut chunked);
        }
        r.flush(&mut chunked);

        assert_eq!(single.len(), chunked.len());
        for (a, b) in single.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn ragged_chunk_sizes_still_match_single_shot() {
        let input = tone(48_000, 500.0, 0.2, 0.3);
        let single = SincResampler::resample_all(48_000, 16_000, &input).expect("build");

        let mut r = SincResampler::new(48_000, 16_000).expect("build");
        let mut chunked = Vec::new();
        let mut i = 0;
        for (n, size) in [7_usize, 1, 960, 13, 480, 2, 1919].iter().cycle().enumerate() {
            let _ = n;
            if i >= input.len() {
                break;
            }
            let end = (i + size).min(input.len());
            r.process_i16(&input[i..end], &mut chunked);
            i = end;
        }
        r.flush(&mut chunked);

        assert_eq!(single.len(), chunked.len());
        for (a, b) in single.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn group_delay_is_compensated_so_legs_stay_aligned() {
        // An impulse at input sample 100 (8 kHz) must land near output
        // sample 200 (16 kHz), not 200 + filter delay.
        let mut input = vec![0_i16; 400];
        input[100] = 32_767;
        let out = SincResampler::resample_all(8_000, 16_000, &input).expect("build");

        let peak = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map_or(0, |(i, _)| i);
        assert!(peak.abs_diff(200) <= 1, "impulse landed at {peak}, want ~200");
    }

    #[test]
    fn empty_input_is_a_no_op() {
        let mut r = SincResampler::new(8_000, 16_000).expect("build");
        let mut out = Vec::new();
        r.process_i16(&[], &mut out);
        assert_eq!(out, [] as [f32; 0]);
        r.flush(&mut out);
        // Flushing an empty stream emits only the zero tail, which is silence.
        assert!(out.iter().all(|s| s.abs() < 1e-6));
    }

    #[test]
    fn history_stays_bounded_over_a_long_stream() {
        let mut r = SincResampler::new(8_000, 16_000).expect("build");
        let frame = vec![1_000_i16; 160];
        let mut out = Vec::new();
        for _ in 0..3_000 {
            // 60 seconds of call audio.
            r.process_i16(&frame, &mut out);
            out.clear();
        }
        assert!(
            r.history.len() < 4 * HALF_TAPS + 8,
            "history grew to {}",
            r.history.len()
        );
    }
}

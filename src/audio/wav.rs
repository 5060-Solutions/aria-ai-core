//! Reading PCM WAV files.
//!
//! Desktop hosts already write call recordings as WAV, so transcribing one is
//! a matter of reading it back rather than tapping live RTP. This is a minimal
//! reader for exactly that case: uncompressed 16-bit PCM, any channel count,
//! averaged to mono.

use crate::error::AiError;
use std::path::Path;

/// Read a WAV file as mono `i16`, returning the samples and their rate.
///
/// Multi-channel files are averaged rather than interleaved or de-interleaved,
/// so a stereo recording comes back at its true duration with both sides
/// audible.
///
/// # Errors
/// [`AiError::Config`] if the file is not RIFF/WAVE, is compressed, or is not
/// 16-bit; [`AiError::Io`] if it cannot be read.
pub fn read_mono_i16(path: &Path) -> Result<(Vec<i16>, u32), AiError> {
    let bytes = std::fs::read(path).map_err(|e| AiError::io(path, &e))?;
    let bad = |what: &str| AiError::Config(format!("{}: {what}", path.display()));

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }

    let u16_at = |o: usize| -> u16 { u16::from_le_bytes([bytes[o], bytes[o + 1]]) };
    let u32_at = |o: usize| -> u32 {
        u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
    };

    let mut pos = 12;
    let mut channels = 0_u16;
    let mut rate = 0_u32;
    let mut bits = 0_u16;
    let mut format = 0_u16;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(pos + 4) as usize;
        let body = pos + 8;
        if body + size > bytes.len() {
            // Some encoders over-declare the final chunk; clamp rather than fail.
            if id == b"data" {
                let samples = decode_pcm(&bytes[body..], bits, channels, &bad)?;
                return Ok((samples, rate));
            }
            break;
        }

        match id {
            b"fmt " if size >= 16 => {
                format = u16_at(body);
                channels = u16_at(body + 2);
                rate = u32_at(body + 4);
                bits = u16_at(body + 14);
            }
            b"data" => {
                if format != 1 {
                    return Err(bad(&format!(
                        "only uncompressed PCM is supported (format {format})"
                    )));
                }
                let samples = decode_pcm(&bytes[body..body + size], bits, channels, &bad)?;
                return Ok((samples, rate));
            }
            _ => {}
        }
        // Chunks are word-aligned.
        pos = body + size + (size & 1);
    }
    Err(bad("no data chunk"))
}

fn decode_pcm(
    data: &[u8],
    bits: u16,
    channels: u16,
    bad: &dyn Fn(&str) -> AiError,
) -> Result<Vec<i16>, AiError> {
    if bits != 16 {
        return Err(bad(&format!("only 16-bit PCM is supported (got {bits})")));
    }
    let channels = usize::from(channels.max(1));
    // Small by construction (a WAV channel count), so this cannot lose data.
    let divisor = i32::try_from(channels).unwrap_or(1);
    let frames = data.len() / 2 / channels;
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        // Average the channels so a stereo recording does not come through at
        // double rate or silently lose a side.
        let mut acc = 0_i32;
        for c in 0..channels {
            let o = (f * channels + c) * 2;
            acc += i32::from(i16::from_le_bytes([data[o], data[o + 1]]));
        }
        out.push(i16::try_from(acc / divisor).unwrap_or(0));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Build a minimal PCM WAV. Test-only, so a length that does not fit a
    /// u32 is a broken test rather than a case to handle.
    fn wav(channels: u16, rate: u32, samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let data_len = u32::try_from(data.len()).expect("test data fits a u32");
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data_len).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16_u32.to_le_bytes());
        v.extend_from_slice(&1_u16.to_le_bytes()); // PCM
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
        v.extend_from_slice(&(channels * 2).to_le_bytes());
        v.extend_from_slice(&16_u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        v.extend_from_slice(&data);
        v
    }

    fn write_temp(bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("a.wav");
        std::fs::write(&p, bytes).expect("write");
        (dir, p)
    }

    #[test]
    fn a_mono_wav_reads_back_sample_for_sample() {
        let samples = [0_i16, 1000, -1000, 32767, -32768];
        let (_d, p) = write_temp(&wav(1, 8000, &samples));
        let (got, rate) = read_mono_i16(&p).expect("read");
        assert_eq!(rate, 8000);
        assert_eq!(got, samples);
    }

    #[test]
    fn a_stereo_wav_is_averaged_to_mono_not_interleaved() {
        // L/R pairs: (100,300) -> 200, (0,0) -> 0, (-100,-300) -> -200
        let interleaved = [100_i16, 300, 0, 0, -100, -300];
        let (_d, p) = write_temp(&wav(2, 16_000, &interleaved));
        let (got, rate) = read_mono_i16(&p).expect("read");
        assert_eq!(rate, 16_000);
        assert_eq!(got, vec![200, 0, -200], "frames, not samples");
    }

    #[test]
    fn a_wav_with_an_extra_chunk_before_data_is_still_read() {
        // Real recorders emit LIST/INFO between fmt and data.
        let mut v = wav(1, 8000, &[7_i16, 8, 9]);
        let data_at = v.windows(4).position(|w| w == b"data").expect("data chunk");
        let extra: Vec<u8> = b"LIST\x04\x00\x00\x00INFO".to_vec();
        v.splice(data_at..data_at, extra.iter().copied());
        let riff_len = u32::try_from(v.len() - 8).expect("fits a u32");
        v[4..8].copy_from_slice(&riff_len.to_le_bytes());

        let (_d, p) = write_temp(&v);
        let (got, _) = read_mono_i16(&p).expect("read");
        assert_eq!(got, vec![7, 8, 9]);
    }

    #[test]
    fn non_wav_and_unsupported_encodings_are_rejected_clearly() {
        let (_d, p) = write_temp(b"this is not a wav");
        assert!(read_mono_i16(&p).is_err());

        // 8-bit PCM: valid RIFF, unsupported depth.
        let mut v = wav(1, 8000, &[1_i16]);
        let bits_at = v.windows(4).position(|w| w == b"WAVE").expect("wave") + 4 + 8 + 14;
        v[bits_at..bits_at + 2].copy_from_slice(&8_u16.to_le_bytes());
        let (_d2, p2) = write_temp(&v);
        let err = read_mono_i16(&p2).expect_err("should reject");
        assert!(format!("{err}").contains("16-bit"), "{err}");
    }
}

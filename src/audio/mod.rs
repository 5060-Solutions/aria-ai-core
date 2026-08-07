//! The audio front end: two call legs in, 16 kHz mono `f32` and a
//! speaker-attributed segment timeline out.
//!
//! Nothing in here needs a model, a file, or a network — it is pure and
//! unit-testable with synthetic audio, which is the point.

pub mod resample;
pub mod ring;
pub mod segment;
pub mod tap;
pub mod vad;
pub mod wav;

pub use resample::{SincResampler, TARGET_RATE_HZ};
pub use ring::{PcmRing, RingStats};
pub use segment::{segment_call, slice_ms, AudioSegment, SegmentationConfig};
pub use tap::{CallAudioTap, CapturedAudio, LegStats, TapConfig};
pub use vad::{detect_speech, rms, total_speech_ms, SpeechSpan, VadConfig};

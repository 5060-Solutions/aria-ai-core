//! Small internal helpers that exist purely so the rest of the crate never
//! has to `unwrap`.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Lock a mutex, tolerating poisoning.
///
/// A poisoned mutex here means some other thread panicked while holding it.
/// Propagating that as a panic across an FFI boundary is undefined behaviour,
/// and every mutex in this crate guards plain data (buffers, counters, maps)
/// that is still structurally valid after a panic. So we take the inner guard
/// and carry on.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Lowercase hex encoding. Avoids pulling in the `hex` crate for 12 lines.
pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from(DIGITS[usize::from(b >> 4)]));
        out.push(char::from(DIGITS[usize::from(b & 0x0f)]));
    }
    out
}

/// Convert a sample count at a known rate into whole milliseconds.
pub const fn samples_to_ms(samples: usize, rate_hz: u32) -> u64 {
    if rate_hz == 0 {
        return 0;
    }
    (samples as u64) * 1000 / (rate_hz as u64)
}

/// Convert milliseconds at a known rate into a sample count.
pub const fn ms_to_samples(ms: u64, rate_hz: u32) -> usize {
    (ms.saturating_mul(rate_hz as u64) / 1000) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_known_vectors() {
        assert_eq!(to_hex(&[]), "");
        assert_eq!(to_hex(&[0x00, 0x0f, 0xf0, 0xff]), "000ff0ff");
        assert_eq!(to_hex(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
    }

    #[test]
    fn sample_time_conversions_are_inverses_on_whole_ms() {
        assert_eq!(samples_to_ms(16_000, 16_000), 1000);
        assert_eq!(samples_to_ms(160, 8_000), 20);
        assert_eq!(ms_to_samples(20, 8_000), 160);
        assert_eq!(ms_to_samples(20, 16_000), 320);
        // Guards against a divide-by-zero panic if a codec reports rate 0.
        assert_eq!(samples_to_ms(100, 0), 0);
    }

    #[test]
    fn poisoned_lock_is_recovered_not_propagated() {
        use std::sync::Arc;
        let m = Arc::new(Mutex::new(7_u32));
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _g = m2.lock();
            panic!("poison it");
        })
        .join();
        assert!(m.is_poisoned());
        assert_eq!(*lock(&m), 7);
    }
}

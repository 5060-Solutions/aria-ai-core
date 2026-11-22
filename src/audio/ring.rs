//! Bounded, drop-oldest PCM ring buffer — one per call leg.
//!
//! The producer is an RTP thread that must never block (design spec §3.4).
//! `push` therefore does a bounded `copy_from_slice` under a mutex that is
//! held for nothing else: no allocation, no I/O, no syscalls, no waiting on a
//! second lock. On overflow it discards the *oldest* samples and bumps a
//! counter rather than blocking or growing.
//!
//! A true lock-free SPSC ring is the eventual goal (spec §3.4), but drop-oldest
//! requires the producer to move the consumer's read cursor, which is exactly
//! the case SPSC does not cover. Since the critical section is a memcpy of at
//! most one RTP frame, the mutex is the honest trade for Phase 1. It is
//! isolated behind this type so it can be swapped without touching callers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::util::lock;

/// Counters describing how a leg's buffer has behaved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct RingStats {
    /// Samples accepted from the producer (including ones later dropped).
    pub pushed_samples: u64,
    /// Oldest samples discarded because the consumer fell behind.
    pub dropped_samples: u64,
    /// Reads that returned fewer samples than asked for (consumer starved).
    pub underruns: u64,
    /// Samples currently buffered.
    pub len: u64,
    /// Buffer capacity in samples.
    pub capacity: u64,
}

impl RingStats {
    /// True if any audio was lost. Surfaced so the UI can say the transcript
    /// may have gaps rather than silently lying.
    #[must_use]
    pub const fn lost_audio(&self) -> bool {
        self.dropped_samples > 0
    }
}

#[derive(Debug)]
struct Inner {
    buf: Box<[i16]>,
    head: usize,
    len: usize,
}

/// Fixed-capacity mono `i16` ring buffer with drop-oldest overflow.
#[derive(Debug)]
pub struct PcmRing {
    inner: Mutex<Inner>,
    capacity: usize,
    pushed: AtomicU64,
    dropped: AtomicU64,
    underruns: AtomicU64,
}

impl PcmRing {
    /// Allocate a ring holding `capacity` samples. A zero capacity is clamped
    /// to 1 so no arithmetic below can divide by zero.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Mutex::new(Inner {
                buf: vec![0_i16; capacity].into_boxed_slice(),
                head: 0,
                len: 0,
            }),
            capacity,
            pushed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
        }
    }

    /// Allocate a ring sized for `ms` milliseconds at `rate_hz`.
    #[must_use]
    pub fn for_duration(ms: u64, rate_hz: u32) -> Self {
        Self::with_capacity(crate::util::ms_to_samples(ms, rate_hz))
    }

    /// Capacity in samples.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Samples currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        lock(&self.inner).len
    }

    /// True when nothing is buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of the counters.
    #[must_use]
    pub fn stats(&self) -> RingStats {
        let len = self.len();
        RingStats {
            pushed_samples: self.pushed.load(Ordering::Relaxed),
            dropped_samples: self.dropped.load(Ordering::Relaxed),
            underruns: self.underruns.load(Ordering::Relaxed),
            len: len as u64,
            capacity: self.capacity as u64,
        }
    }

    /// Write `samples`, discarding the oldest data if there is not room.
    ///
    /// Returns how many *old* samples were discarded. Never blocks on
    /// anything but the memcpy, never allocates, never fails.
    pub fn push(&self, samples: &[i16]) -> usize {
        if samples.is_empty() {
            return 0;
        }
        self.pushed
            .fetch_add(samples.len() as u64, Ordering::Relaxed);

        let cap = self.capacity;
        let mut inner = lock(&self.inner);

        // Input larger than the whole ring: keep only its tail.
        let (src, extra_dropped) = if samples.len() > cap {
            let skip = samples.len() - cap;
            (&samples[skip..], skip)
        } else {
            (samples, 0)
        };

        let overflow = (inner.len + src.len()).saturating_sub(cap);
        if overflow > 0 {
            inner.head = (inner.head + overflow) % cap;
            inner.len -= overflow;
        }

        let mut write = (inner.head + inner.len) % cap;
        for &s in src {
            // `write` is always < cap by construction.
            if let Some(slot) = inner.buf.get_mut(write) {
                *slot = s;
            }
            write = if write + 1 == cap { 0 } else { write + 1 };
        }
        inner.len = (inner.len + src.len()).min(cap);
        drop(inner);

        let dropped = overflow + extra_dropped;
        if dropped > 0 {
            self.dropped.fetch_add(dropped as u64, Ordering::Relaxed);
        }
        dropped
    }

    /// Move up to `max` samples out of the ring, appending to `out`.
    ///
    /// Returns the number of samples moved. If fewer than `max` were
    /// available *and* the ring was not empty to begin with, an underrun is
    /// recorded — that is the consumer asking for more than the producer has
    /// delivered, which is normal at end of call and a warning sign mid-call.
    pub fn drain_into(&self, out: &mut Vec<i16>, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        let cap = self.capacity;
        let mut inner = lock(&self.inner);
        let n = inner.len.min(max);
        if n < max {
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
        if n == 0 {
            return 0;
        }
        out.reserve(n);
        let head = inner.head;
        for i in 0..n {
            let idx = (head + i) % cap;
            out.push(inner.buf.get(idx).copied().unwrap_or(0));
        }
        inner.head = (head + n) % cap;
        inner.len -= n;
        n
    }

    /// Move everything currently buffered into `out`.
    pub fn drain_all(&self, out: &mut Vec<i16>) -> usize {
        let max = self.len();
        self.drain_into(out, max)
    }

    /// Discard everything buffered without counting it as dropped.
    pub fn clear(&self) {
        let mut inner = lock(&self.inner);
        inner.head = 0;
        inner.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_then_drain_round_trips_in_order() {
        let r = PcmRing::with_capacity(16);
        assert_eq!(r.push(&[1, 2, 3]), 0);
        assert_eq!(r.push(&[4, 5]), 0);
        let mut out = Vec::new();
        assert_eq!(r.drain_all(&mut out), 5);
        assert_eq!(out, vec![1, 2, 3, 4, 5]);
        assert!(r.is_empty());
    }

    #[test]
    fn wraps_around_correctly() {
        let r = PcmRing::with_capacity(4);
        r.push(&[1, 2, 3]);
        let mut out = Vec::new();
        r.drain_into(&mut out, 2); // head now at 2
        r.push(&[4, 5, 6]); // wraps
        out.clear();
        assert_eq!(r.drain_all(&mut out), 4);
        assert_eq!(out, vec![3, 4, 5, 6]);
    }

    #[test]
    fn overrun_drops_the_oldest_samples_and_counts_them() {
        let r = PcmRing::with_capacity(4);
        assert_eq!(r.push(&[1, 2, 3, 4]), 0);
        // Two more: 1 and 2 must fall off the front.
        assert_eq!(r.push(&[5, 6]), 2);

        let mut out = Vec::new();
        r.drain_all(&mut out);
        assert_eq!(out, vec![3, 4, 5, 6]);

        let s = r.stats();
        assert_eq!(s.pushed_samples, 6);
        assert_eq!(s.dropped_samples, 2);
        assert!(s.lost_audio());
    }

    #[test]
    fn a_push_larger_than_the_whole_ring_keeps_only_the_tail() {
        let r = PcmRing::with_capacity(3);
        let dropped = r.push(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(dropped, 4);
        let mut out = Vec::new();
        r.drain_all(&mut out);
        assert_eq!(out, vec![5, 6, 7]);
        assert_eq!(r.stats().dropped_samples, 4);
    }

    #[test]
    fn underrun_returns_what_is_there_and_is_counted() {
        let r = PcmRing::with_capacity(64);
        r.push(&[9, 9, 9]);
        let mut out = Vec::new();
        assert_eq!(r.drain_into(&mut out, 100), 3);
        assert_eq!(out.len(), 3);
        assert_eq!(r.stats().underruns, 1);

        // Draining an empty ring is an underrun, not a panic.
        out.clear();
        assert_eq!(r.drain_into(&mut out, 10), 0);
        assert_eq!(out, [] as [i16; 0]);
        assert_eq!(r.stats().underruns, 2);
    }

    #[test]
    fn zero_length_operations_are_no_ops() {
        let r = PcmRing::with_capacity(8);
        assert_eq!(r.push(&[]), 0);
        let mut out = Vec::new();
        assert_eq!(r.drain_into(&mut out, 0), 0);
        assert_eq!(r.stats().pushed_samples, 0);
        assert_eq!(r.stats().underruns, 0);
    }

    #[test]
    fn zero_capacity_is_clamped_not_fatal() {
        let r = PcmRing::with_capacity(0);
        assert_eq!(r.capacity(), 1);
        r.push(&[1, 2, 3]);
        let mut out = Vec::new();
        r.drain_all(&mut out);
        assert_eq!(out, vec![3]);
    }

    #[test]
    fn duration_sizing_matches_the_rtp_frame_grid() {
        // 5 seconds of 8 kHz telephony audio.
        let r = PcmRing::for_duration(5_000, 8_000);
        assert_eq!(r.capacity(), 40_000);
    }

    #[test]
    fn concurrent_producer_and_consumer_never_lose_ordering() {
        use std::sync::Arc;

        let r = Arc::new(PcmRing::with_capacity(4_096));
        let producer = {
            let r = Arc::clone(&r);
            std::thread::spawn(move || {
                for i in 0..10_000_i16 {
                    r.push(&[i]);
                }
            })
        };

        let mut seen: Vec<i16> = Vec::new();
        let mut buf = Vec::new();
        while !producer.is_finished() || !r.is_empty() {
            buf.clear();
            r.drain_all(&mut buf);
            seen.extend_from_slice(&buf);
        }
        let _ = producer.join();
        buf.clear();
        r.drain_all(&mut buf);
        seen.extend_from_slice(&buf);

        // Whatever survived must be strictly increasing: drop-oldest removes
        // a prefix, it never reorders.
        assert!(seen.windows(2).all(|w| w[1] > w[0]));
        assert_ne!(seen, [] as [i16; 0]);
    }
}

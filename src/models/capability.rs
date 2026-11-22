//! Device capability gating (design spec §6.2).
//!
//! The core never asks the OS anything — the host fills in a small struct and
//! hands it over. That keeps the crate free of platform APIs and makes every
//! tier decision unit-testable.
//!
//! iOS note for the binding layer: the number that belongs in
//! `available_ram_mb` is `os_proc_available_memory()`, **not**
//! `ProcessInfo.physicalMemory`. `iOS` jetsams well below physical RAM and a
//! `VoIP` app killed mid-call is a catastrophic failure.

use serde::{Deserialize, Serialize};

/// Which OS family. Only used for the desktop/mobile distinction and for
/// deciding whether a GPU backend is plausible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    Ios,
    Android,
    MacOs,
    Windows,
    Linux,
    Other,
}

impl Platform {
    /// Desktop platforms get the generous tier and the extra affordances
    /// (storage-location picker, "install anyway").
    #[must_use]
    pub const fn is_desktop(self) -> bool {
        matches!(self, Self::MacOs | Self::Windows | Self::Linux)
    }
}

/// Coarse device class. Ordered: `Unsupported < Low < Medium < High < Desktop`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[serde(rename_all = "lowercase")]
pub enum DeviceTier {
    /// 32-bit, or too little RAM. Feature hidden with an explanation.
    Unsupported,
    /// STT only, small models.
    Low,
    /// STT up to `small`; LLM up to ~1.7B.
    Medium,
    /// Everything mobile can run.
    High,
    /// No meaningful ceiling.
    Desktop,
}

impl DeviceTier {
    /// Human-readable label for error messages and greyed-out rows.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Desktop => "desktop",
        }
    }
}

/// What the host tells the core about this device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct DeviceCapability {
    pub platform: Platform,
    /// Physical RAM.
    pub total_ram_mb: u32,
    /// RAM this process may realistically use right now. iOS:
    /// `os_proc_available_memory()`. Zero means "unknown, fall back to half
    /// of `total_ram_mb`".
    pub available_ram_mb: u32,
    pub cpu_cores: u32,
    /// False for `armeabi-v7a`. 32-bit devices are excluded entirely.
    pub is_64bit: bool,
    /// Metal / Vulkan available. Informational; does not change the tier.
    pub has_gpu_backend: bool,
}

impl DeviceCapability {
    /// A conservative desktop default, useful for tests and for the CLI.
    #[must_use]
    pub const fn desktop_default() -> Self {
        Self {
            platform: Platform::MacOs,
            total_ram_mb: 16_384,
            available_ram_mb: 8_192,
            cpu_cores: 8,
            is_64bit: true,
            has_gpu_backend: true,
        }
    }

    /// Classify the device (design spec §6.2).
    #[must_use]
    pub const fn tier(&self) -> DeviceTier {
        if !self.is_64bit || self.total_ram_mb < 3_072 {
            return DeviceTier::Unsupported;
        }
        if self.platform.is_desktop() && self.total_ram_mb >= 8_192 {
            return DeviceTier::Desktop;
        }
        if self.total_ram_mb >= 8_192 {
            DeviceTier::High
        } else if self.total_ram_mb >= 6_144 {
            DeviceTier::Medium
        } else {
            DeviceTier::Low
        }
    }

    /// The most RAM a single model may claim: half of what is actually
    /// available to the process.
    #[must_use]
    pub const fn ram_budget_mb(&self) -> u32 {
        let available = if self.available_ram_mb == 0 {
            self.total_ram_mb / 2
        } else {
            self.available_ram_mb
        };
        available / 2
    }

    /// Threads to give an inference run. Never saturate all cores on mobile —
    /// that is what makes the device hot and the UI stutter (design spec §7.3).
    #[must_use]
    pub const fn inference_threads(&self) -> u32 {
        let cores = if self.cpu_cores == 0 {
            2
        } else {
            self.cpu_cores
        };
        if self.platform.is_desktop() {
            if cores > 2 {
                cores - 2
            } else {
                1
            }
        } else if cores > 4 {
            4
        } else {
            cores
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(total: u32, is_64: bool, platform: Platform) -> DeviceCapability {
        DeviceCapability {
            platform,
            total_ram_mb: total,
            available_ram_mb: 0,
            cpu_cores: 8,
            is_64bit: is_64,
            has_gpu_backend: false,
        }
    }

    #[test]
    fn thirty_two_bit_is_always_unsupported() {
        assert_eq!(
            dev(16_384, false, Platform::Android).tier(),
            DeviceTier::Unsupported
        );
    }

    #[test]
    fn tier_boundaries_match_the_spec_table() {
        assert_eq!(
            dev(2_048, true, Platform::Ios).tier(),
            DeviceTier::Unsupported
        );
        assert_eq!(dev(3_072, true, Platform::Ios).tier(), DeviceTier::Low);
        assert_eq!(dev(6_143, true, Platform::Ios).tier(), DeviceTier::Low);
        assert_eq!(dev(6_144, true, Platform::Ios).tier(), DeviceTier::Medium);
        assert_eq!(dev(8_192, true, Platform::Ios).tier(), DeviceTier::High);
        assert_eq!(
            dev(8_192, true, Platform::MacOs).tier(),
            DeviceTier::Desktop
        );
        // A small desktop still gets mobile-grade gating.
        assert_eq!(dev(6_144, true, Platform::Linux).tier(), DeviceTier::Medium);
    }

    #[test]
    fn tiers_are_ordered_for_gating_comparisons() {
        assert!(DeviceTier::Desktop > DeviceTier::High);
        assert!(DeviceTier::High > DeviceTier::Medium);
        assert!(DeviceTier::Low > DeviceTier::Unsupported);
    }

    #[test]
    fn ram_budget_prefers_available_over_physical() {
        let mut d = dev(8_192, true, Platform::Ios);
        assert_eq!(d.ram_budget_mb(), 2_048); // unknown => total/2 then /2
        d.available_ram_mb = 1_400; // typical iOS jetsam headroom
        assert_eq!(d.ram_budget_mb(), 700);
    }

    #[test]
    fn thread_budget_never_saturates_a_phone() {
        let mut d = dev(8_192, true, Platform::Android);
        d.cpu_cores = 8;
        assert_eq!(d.inference_threads(), 4);
        d.cpu_cores = 2;
        assert_eq!(d.inference_threads(), 2);
        d.cpu_cores = 0;
        assert_eq!(d.inference_threads(), 2);

        let mut m = dev(16_384, true, Platform::MacOs);
        m.cpu_cores = 10;
        assert_eq!(m.inference_threads(), 8);
        m.cpu_cores = 1;
        assert_eq!(m.inference_threads(), 1);
    }
}

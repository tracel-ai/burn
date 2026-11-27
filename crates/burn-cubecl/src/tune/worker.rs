//! What one worker of a device holds at once ([`WorkerLimits`]): the facts a prediction needs to
//! know how many cubes run side by side, which no runtime reports.

use cubek::std::Hardware;

/// What one worker holds at once — one of the `num_streaming_multiprocessors` its runtime
/// reports, whatever the vendor calls it — and how many units issue arithmetic on it each cycle.
///
/// No runtime reports these, so they are read off the architecture the device compiles for: a
/// table of the ones this crate has numbers for, and nothing for any other. A prediction made
/// without them knows only what the runtime reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerLimits {
    /// Units resident at once.
    pub units_count: u32,
    /// Cubes resident at once, whatever their size.
    pub cubes_count: u32,
    /// 32-bit registers its units share.
    pub registers_count: u32,
    /// Shared memory its cubes share.
    pub shared_memory_bytes: usize,
    /// Units that issue vector arithmetic each cycle.
    pub lanes_count: u32,
}

impl WorkerLimits {
    /// The limits of a worker of `hw`'s architecture, read off the target it compiles for, or
    /// `None` for an architecture this table has no numbers for.
    pub fn of(hw: &Hardware) -> Option<Self> {
        Self::of_target(&hw.properties.identity.fingerprint)
    }

    /// The limits for the compile target `fingerprint` names (`hip-kernel_gfx1151`, `ptx_sm90`).
    fn of_target(fingerprint: &str) -> Option<Self> {
        if let Some(arch) = fingerprint.strip_prefix("hip-kernel_gfx11") {
            return Some(Self::rdna3(arch));
        }
        let sm: u32 = fingerprint.strip_prefix("ptx_sm")?.parse().ok()?;
        Self::nvidia(sm)
    }

    /// An RDNA3 or RDNA3.5 work-group processor: the worker HIP counts, two compute units of
    /// two wave32 SIMDs each, sixteen waves a SIMD, 64 KB of LDS a compute unit. The register
    /// file is 1536 registers a lane on Navi 31 and 32 (`gfx1100`, `gfx1101`), 1024 on the rest.
    fn rdna3(arch: &str) -> Self {
        let registers_per_lane = match arch.starts_with("00") || arch.starts_with("01") {
            true => 1536,
            false => 1024,
        };
        const SIMDS: u32 = 4;
        const WAVE: u32 = 32;
        Self {
            units_count: SIMDS * 16 * WAVE,
            cubes_count: 32,
            registers_count: SIMDS * registers_per_lane * WAVE,
            shared_memory_bytes: 128 * 1024,
            lanes_count: SIMDS * WAVE,
        }
    }

    /// An NVIDIA streaming multiprocessor of compute capability `sm`, as the CUDA occupancy
    /// tables state it, or `None` for one they are not written down for here.
    fn nvidia(sm: u32) -> Option<Self> {
        let (units_count, cubes_count, shared_kb, lanes_count) = match sm {
            80 => (2048, 32, 164, 64),
            86 | 87 => (1536, 16, 100, 128),
            89 => (1536, 24, 100, 128),
            90 => (2048, 32, 228, 128),
            _ => return None,
        };
        Some(Self {
            units_count,
            cubes_count,
            registers_count: 64 * 1024,
            shared_memory_bytes: shared_kb * 1024,
            lanes_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdna3_is_read_off_the_hip_target() {
        let halo = WorkerLimits::of_target("hip-kernel_gfx1151").expect("rdna3.5");
        let navi31 = WorkerLimits::of_target("hip-kernel_gfx1100").expect("rdna3");
        assert_eq!(halo.units_count, 2048);
        assert!(navi31.registers_count > halo.registers_count);
    }

    #[test]
    fn nvidia_is_read_off_the_ptx_target() {
        assert_eq!(
            WorkerLimits::of_target("ptx_sm90").map(|limits| limits.units_count),
            Some(2048)
        );
        assert_eq!(
            WorkerLimits::of_target("ptx_sm86").map(|limits| limits.cubes_count),
            Some(16)
        );
    }

    #[test]
    fn an_architecture_without_numbers_has_no_limits() {
        assert_eq!(WorkerLimits::of_target("ptx_sm75"), None);
        assert_eq!(WorkerLimits::of_target("hip-kernel_gfx942"), None);
        assert_eq!(WorkerLimits::of_target("spirv"), None);
    }
}

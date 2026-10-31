//! What a device is doing, or can do: the answers that have to run something —
//! read its counters, look through its allocator, launch probes.

use burn_std::DType;

/// One reading of the counters the card behind a device keeps: how busy it and
/// its memory are, how hot it runs and what it draws. A figure is `None` where
/// no counter gives it, and for a counter that measures between two readings,
/// the first time it is read.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeviceTelemetry {
    /// The share of the counter's last period the device spent running work,
    /// from 0 to 100.
    pub busy_percent: Option<f32>,
    /// The share of the counter's last period its memory spent reading or
    /// writing, from 0 to 100.
    pub memory_busy_percent: Option<f32>,
    /// How hot it runs, in degrees Celsius.
    pub temperature_celsius: Option<f32>,
    /// What it draws, in watts.
    pub power_watts: Option<f32>,
}

/// A peak throughput a device is asked for: what a probe launched on it
/// measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PeakKey {
    /// Traffic across the memory interface, over as much as it takes at once,
    /// in bytes per second.
    Memory(MemoryAccess),
    /// Arithmetic on operands of `dtype` without matrix hardware, in
    /// operations per second.
    Compute {
        /// The operands' type.
        dtype: DType,
    },
    /// A matrix product on the device's matrix hardware, operands of `dtype`
    /// summed in `accumulator`, one `tile` at a time, in operations per second.
    Matmul {
        /// The operands' type.
        dtype: DType,
        /// The running sum's type.
        accumulator: DType,
        /// The shape the hardware multiplies at once.
        tile: MatmulTile,
    },
    /// Kernel launches, in launches per second.
    Launch,
}

/// Which directions of traffic a [`PeakKey::Memory`] probe issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MemoryAccess {
    /// Reads every byte and writes it back: the ceiling of a kernel that both
    /// loads and stores.
    Copy,
    /// Reads only: the ceiling of a kernel that streams what it never writes
    /// back, which may exceed a copy's.
    Read,
    /// Writes only: the ceiling of a kernel that streams stores it never reads
    /// back.
    Write,
}

/// The shape a device's matrix hardware multiplies at once: an `m`×`k` by
/// `k`×`n` product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MatmulTile {
    /// Rows of the product.
    pub m: u32,
    /// Columns of the product.
    pub n: u32,
    /// The inner dimension.
    pub k: u32,
}

/// A peak throughput, per second, in the unit its [`PeakKey`] names.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Peak {
    /// Bytes, operations or launches per second.
    pub per_second: f64,
}

/// Why a device has no peak for a [`PeakKey`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PeakError {
    /// The device implements no such operation, or the backend runs no probe.
    Unsupported,
    /// The device's timer reported no elapsed time for any shape of the probe.
    NoTiming,
    /// The device could not allocate the buffers the probe runs over.
    Allocation,
    /// The probe did not run: it failed to compile or launch, or the device
    /// faulted under it.
    Launch,
}

impl core::fmt::Display for PeakError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Unsupported => "unsupported",
            Self::NoTiming => "no timing",
            Self::Allocation => "allocation failed",
            Self::Launch => "launch failed",
        })
    }
}

impl core::error::Error for PeakError {}

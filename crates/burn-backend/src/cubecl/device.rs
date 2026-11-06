//! A device's description and measurements in cubecl's words, for a backend
//! that answers them on cubecl.

use crate::{DeviceDescription, HardwareKind, MemoryAccess, PeakError, PeakKey};
use cubecl::device as cube_device;
use cubecl::throughput as cube;

impl From<cube_device::DeviceDescription> for DeviceDescription {
    fn from(description: cube_device::DeviceDescription) -> Self {
        Self {
            name: description.name,
            kind: description.kind.into(),
            max_memory: description.max_memory,
            architecture: description.architecture,
            driver_version: description.driver_version,
        }
    }
}

impl From<cube_device::DeviceKind> for HardwareKind {
    fn from(kind: cube_device::DeviceKind) -> Self {
        match kind {
            cube_device::DeviceKind::Discrete => Self::DiscreteGpu,
            cube_device::DeviceKind::Integrated => Self::IntegratedGpu,
            cube_device::DeviceKind::Virtual => Self::VirtualGpu,
            cube_device::DeviceKind::Cpu => Self::Cpu,
            // The kind is non-exhaustive: a kind cubecl adds later is none of
            // the above to Burn until it names it.
            _ => Self::Other,
        }
    }
}

impl From<PeakKey> for cube::ThroughputKey {
    fn from(key: PeakKey) -> Self {
        let mode = match key {
            PeakKey::Memory(access) => cube::ThroughputMode::memory(access.into()),
            PeakKey::Compute { dtype } => cube::ThroughputMode::ComputeDirect {
                dtype: super::dtype_to_elem_type(dtype),
            },
            PeakKey::Matmul {
                dtype,
                accumulator,
                tile,
            } => cube::ThroughputMode::ComputeCmma {
                dtype: super::dtype_to_elem_type(dtype),
                config: cube::ComputeCmmaConfig {
                    accumulator_type: super::dtype_to_elem_type(accumulator),
                    cmma_dims: cube::CmmaDims {
                        m: tile.m as usize,
                        n: tile.n as usize,
                        k: tile.k as usize,
                    },
                },
            },
            PeakKey::Launch => cube::ThroughputMode::Launch,
        };
        Self { mode }
    }
}

impl From<MemoryAccess> for cube::MemoryAccess {
    fn from(access: MemoryAccess) -> Self {
        match access {
            MemoryAccess::Copy => Self::Copy,
            MemoryAccess::Read => Self::Read,
            MemoryAccess::Write => Self::Write,
        }
    }
}

impl From<cube::ThroughputError> for PeakError {
    fn from(error: cube::ThroughputError) -> Self {
        match error {
            cube::ThroughputError::Unsupported => Self::Unsupported,
            cube::ThroughputError::NoTiming => Self::NoTiming,
            cube::ThroughputError::Allocation => Self::Allocation,
            cube::ThroughputError::Launch => Self::Launch,
        }
    }
}

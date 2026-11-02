//! What a device is, as its driver states it: the answers that cost nothing to
//! run.

use alloc::string::String;

/// What a device is, as its driver states it before anything runs on it:
/// enough to list a machine's devices, show them to a person and pick one.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DeviceDescription {
    /// The name the device reports, e.g. `NVIDIA GeForce RTX 5070 Laptop GPU`.
    pub name: String,
    /// The sort of hardware it is.
    pub kind: HardwareKind,
    /// The memory the device can use, in bytes: the card's own for a GPU, the
    /// machine's for a CPU, and on Metal the working set the device recommends
    /// staying under. `None` where the driver does not say.
    pub max_memory: Option<u64>,
    /// The architecture its kernels are compiled for, e.g. `sm_90` or
    /// `gfx1151`. `None` where the driver does not say.
    pub architecture: Option<String>,
    /// The driver's version, in its own API's terms. `None` where the driver
    /// does not say.
    pub driver_version: Option<String>,
}

/// The sort of hardware a [`DeviceDescription`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HardwareKind {
    /// A GPU with memory of its own.
    DiscreteGpu,
    /// A GPU sharing the host's memory: an APU, Apple silicon, an Intel iGPU.
    IntegratedGpu,
    /// A GPU a hypervisor hands to a virtual machine.
    VirtualGpu,
    /// The host's processors, or a software rasterizer running on them.
    Cpu,
    /// Whatever the driver calls none of the above.
    Other,
}

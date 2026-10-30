use burn::prelude::*;
#[cfg(any(feature = "cuda", feature = "rocm", feature = "wgpu"))]
use burn::tensor::DeviceType;

/// One device per card, whichever runtime reaches it best.
#[cfg(any(feature = "cuda", feature = "rocm", feature = "wgpu"))]
pub fn available() -> Vec<Device> {
    let preference = runtimes_in_order();

    Device::enumerate_physical()
        .into_iter()
        .map(|gpu| preferred(gpu.devices.into_vec(), &preference))
        .collect()
}

/// A card several runtimes reach is taken through the first of these that reaches it, so a native
/// runtime wins over a portable one.
#[cfg(any(feature = "cuda", feature = "rocm", feature = "wgpu"))]
fn runtimes_in_order() -> Vec<Vec<Device>> {
    let kinds: &[DeviceType] = &[
        #[cfg(feature = "cuda")]
        DeviceType::Cuda,
        #[cfg(feature = "rocm")]
        DeviceType::Rocm,
        #[cfg(feature = "metal")]
        DeviceType::Metal,
        #[cfg(feature = "vulkan")]
        DeviceType::Vulkan,
        #[cfg(feature = "webgpu")]
        DeviceType::WebGpu,
    ];

    kinds
        .iter()
        .map(|kind| Device::enumerate(kind.clone()).into_vec())
        .collect()
}

/// A card no listed runtime enumerated keeps whatever device `enumerate_physical` gave it.
#[cfg(any(feature = "cuda", feature = "rocm", feature = "wgpu"))]
fn preferred(devices: Vec<Device>, preference: &[Vec<Device>]) -> Device {
    preference
        .iter()
        .find_map(|reachable| devices.iter().find(|device| reachable.contains(device)))
        .unwrap_or(&devices[0])
        .clone()
}

/// Two CPU backends, so the split is a real one on a machine with no GPU.
#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "wgpu")))]
pub fn available() -> Vec<Device> {
    vec![Device::flex(), Device::cpu()]
}

/// The devices to train across.
#[cfg(any(feature = "cuda", feature = "rocm", feature = "wgpu"))]
pub fn trainable() -> Vec<Device> {
    available()
}

/// Flex alone: the CPU backend forwards but cannot launch the backward matmul, so training across
/// the two CPU backends would panic partway through the first step.
#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "wgpu")))]
pub fn trainable() -> Vec<Device> {
    vec![Device::flex()]
}

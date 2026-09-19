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
    #[allow(unused_mut)]
    let mut kinds: Vec<DeviceType> = Vec::new();
    #[cfg(feature = "cuda")]
    kinds.push(DeviceType::Cuda);
    #[cfg(feature = "rocm")]
    kinds.push(DeviceType::Rocm);
    #[cfg(feature = "metal")]
    kinds.push(DeviceType::Metal);
    #[cfg(feature = "vulkan")]
    kinds.push(DeviceType::Vulkan);
    #[cfg(feature = "webgpu")]
    kinds.push(DeviceType::WebGpu);

    kinds
        .into_iter()
        .map(|kind| Device::enumerate(kind).into_vec())
        .collect()
}

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

#![recursion_limit = "256"]

use burn::tensor::Device;
use mnist::training;

#[allow(unreachable_code)]
fn select_device() -> Device {
    #[cfg(feature = "flex")]
    return Device::flex();

    #[cfg(all(feature = "tch-gpu", not(target_os = "macos")))]
    return Device::libtorch_cuda(burn::tensor::DeviceIndex::Default);

    #[cfg(all(feature = "tch-gpu", target_os = "macos"))]
    return Device::libtorch_mps();

    #[cfg(feature = "tch-cpu")]
    return Device::libtorch();

    #[cfg(feature = "vulkan")]
    return Device::vulkan(burn::tensor::DeviceKind::DefaultDevice);
    #[cfg(feature = "metal")]
    return Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    #[cfg(feature = "wgpu")]
    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);

    #[cfg(feature = "cuda")]
    return Device::cuda(burn::tensor::DeviceIndex::Default);

    #[cfg(feature = "rocm")]
    return Device::rocm(burn::tensor::DeviceIndex::Default);

    #[cfg(feature = "remote")]
    return Device::remote("ws://localhost:3000", 0);

    unreachable!("At least one backend will be selected.")
}

fn main() {
    let device = select_device();
    training::run(device);
}

use burn::tensor::Device;

#[allow(unreachable_code)]
fn select_device() -> Device {
    #[cfg(not(any(
        feature = "wgpu",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm",
        feature = "cuda",
    )))]
    return Device::flex();

    #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);

    #[cfg(feature = "cuda")]
    return Device::cuda(burn::tensor::DeviceIndex::Default);

    #[cfg(feature = "rocm")]
    return Device::rocm(burn::tensor::DeviceIndex::Default);

    unreachable!("At least one backend will be selected.")
}

fn main() {
    let device = select_device();
    dqn_agent::training::run(device.autodiff());
}

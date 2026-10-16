use burn::tensor::Device;
use mnist::training;

#[allow(unreachable_code)]
fn select_device() -> Device {
    #[cfg(feature = "flex")]
    return Device::flex();

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

    unreachable!("At least one backend will be selected.")
}

fn main() {
    let device = select_device();
    training::run(device);
}

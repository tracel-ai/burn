use burn::tensor::Device;
#[cfg(feature = "cuda")]
use burn::tensor::DeviceType;

fn main() {
    pipeline_parallel::run(devices());
}

/// One device per GPU, through CUDA on the cards CUDA reaches.
#[cfg(any(feature = "cuda", feature = "vulkan"))]
fn devices() -> Vec<Device> {
    #[cfg(feature = "cuda")]
    let cuda = Device::enumerate(DeviceType::Cuda);

    Device::enumerate_physical()
        .into_iter()
        .map(|gpu| {
            #[cfg(feature = "cuda")]
            if let Some(device) = gpu.devices.iter().find(|device| cuda.contains(device)) {
                return device.clone();
            }
            gpu.devices[0].clone()
        })
        .collect()
}

/// One CPU device standing in for two, so the split runs on any machine.
#[cfg(not(any(feature = "cuda", feature = "vulkan")))]
fn devices() -> Vec<Device> {
    vec![Device::flex(), Device::flex()]
}

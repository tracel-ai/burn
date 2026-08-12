#![recursion_limit = "256"]

use burn::{
    optim::decay::WeightDecayConfig,
    tensor::{Device, DeviceConfig, Element},
};
use text_generation::{DbPediaDataset, training::ExperimentConfig};

#[cfg(not(any(feature = "f16", feature = "flex32")))]
#[allow(unused)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;
#[cfg(feature = "flex32")]
type ElemType = burn::tensor::flex32;

pub fn launch(mut device: Device) {
    device
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
        .unwrap();

    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    text_generation::training::train::<DbPediaDataset>(
        device,
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::tensor::{Device, DeviceIndex};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = Device::libtorch_cuda(DeviceIndex::Default);
        #[cfg(target_os = "macos")]
        let device = Device::libtorch_mps();

        crate::launch(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::tensor::Device;

    pub fn run() {
        crate::launch(Device::libtorch());
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
mod wgpu {
    use burn::tensor::{Device, DeviceKind};

    pub fn run() {
        crate::launch(Device::wgpu(DeviceKind::DefaultDevice));
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::tensor::{Device, DeviceIndex};

    pub fn run() {
        crate::launch(Device::cuda(DeviceIndex::Default));
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use burn::tensor::{Device, DeviceIndex};

    pub fn run() {
        crate::launch(Device::rocm(DeviceIndex::Default));
    }
}

#[cfg(feature = "flex")]
mod flex {
    use burn::tensor::Device;

    pub fn run() {
        crate::launch(Device::flex());
    }
}

#[cfg(feature = "remote")]
mod remote {
    use burn::tensor::{Device, DeviceType};

    /// Address of the `burn-remote` server to train against.
    const ADDRESS: &str = "ws://localhost:3000";

    /// Train on a single one of the devices the remote server hosts.
    ///
    /// `launch` configures the device it receives, so don't configure the enumerated set here
    /// too — doing both locks the device's settings twice and returns
    /// [`DeviceError::AlreadyInitialized`](burn::tensor::DeviceError::AlreadyInitialized).
    pub fn run() {
        let devices = Device::enumerate(DeviceType::remote_websocket(ADDRESS));
        crate::launch(devices.into_vec().pop().unwrap());
    }
}

fn main() {
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "flex")]
    flex::run();
    #[cfg(feature = "remote")]
    remote::run();
}

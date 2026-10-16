use burn::{optim::RmsPropConfig, tensor::Device};

use wgan::{model::ModelConfig, training::TrainingConfig};

pub fn launch(device: Device) {
    let config = TrainingConfig::new(
        ModelConfig::new(),
        RmsPropConfig::new()
            .with_alpha(0.99)
            .with_momentum(0.0)
            .with_epsilon(0.00000008)
            .with_weight_decay(None)
            .with_centered(false),
    );

    wgan::training::train("/tmp/wgan-mnist", config, device);
}

#[cfg(feature = "flex")]
mod flex {
    use burn::tensor::Device;

    pub fn run() {
        crate::launch(Device::flex());
    }
}

#[cfg(feature = "wgpu")]
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

fn main() {
    #[cfg(feature = "flex")]
    flex::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
}

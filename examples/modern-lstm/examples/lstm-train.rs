use burn::{grad_clipping::GradientClippingConfig, optim::AdamConfig, tensor::Device};
use modern_lstm::{model::LstmNetworkConfig, training::TrainingConfig};

pub fn launch(device: Device) {
    let config = TrainingConfig::new(
        LstmNetworkConfig::new(),
        // Gradient clipping via optimizer config
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
    );

    modern_lstm::training::train("/tmp/modern-lstm", config, device);
}

#[cfg(feature = "flex")]
mod flex {
    use burn::tensor::Device;

    pub fn run() {
        crate::launch(Device::flex());
    }
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
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
}

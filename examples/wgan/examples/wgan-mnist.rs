use burn::{optim::RmsPropConfig, tensor::Device};

use wgan::{model::ModelConfig, training::TrainingConfig};

pub fn launch(device: impl Into<Device>) {
    let config = TrainingConfig::new(
        ModelConfig::new(),
        RmsPropConfig::new()
            .with_alpha(0.99)
            .with_momentum(0.0)
            .with_epsilon(0.00000008)
            .with_weight_decay(None)
            .with_centered(false),
    );

    wgan::training::train("/tmp/wgan-mnist", config, device.into());
}

#[cfg(feature = "flex")]
mod flex {
    use burn::backend::flex::FlexDevice;

    pub fn run() {
        crate::launch(FlexDevice);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::libtorch::LibTorchDevice;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        crate::launch(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::libtorch::LibTorchDevice;

    pub fn run() {
        crate::launch(LibTorchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::wgpu::WgpuDevice;

    pub fn run() {
        crate::launch(WgpuDevice::default());
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::backend::cuda::CudaDevice;

    pub fn run() {
        crate::launch(CudaDevice::default());
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

#![recursion_limit = "256"]

use burn::optim::{SgdConfig, momentum::MomentumConfig};
use custom_image_dataset::training::TrainingConfig;

// Import only when backend features are enabled
#[cfg(any(feature = "tch-gpu", feature = "wgpu", feature = "metal"))]
use {burn::backend::Autodiff, custom_image_dataset::training::train};

/// Creates a training configuration with SGD optimizer and momentum.
fn create_config() -> TrainingConfig {
    TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
        momentum: 0.9,
        dampening: 0.,
        nesterov: false,
    })))
}

fn main() {
    #[allow(unused_variables)]
    let config = create_config();

    #[cfg(feature = "tch-gpu")]
    {
        use burn::backend::libtorch::{LibTorch, LibTorchDevice};

        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        train::<Autodiff<LibTorch>>(config, device);
    }

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        train::<Autodiff<Wgpu>>(config, WgpuDevice::default());
    }

    #[cfg(feature = "metal")]
    {
        // Note: Metal backend may have shader compilation issues on Intel Macs with AMD GPUs
        // If you encounter errors, use WGPU backend as an alternative
        use burn::backend::wgpu::{Metal, WgpuDevice};
        train::<Autodiff<Metal>>(config, WgpuDevice::default());
    }
}

use burn::{
    backend::Autodiff,
    optim::{SgdConfig, momentum::MomentumConfig},
};
use custom_image_dataset::training::{TrainingConfig, train};

fn create_config() -> TrainingConfig {
    TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
        momentum: 0.9,
        dampening: 0.,
        nesterov: false,
    })))
}

fn main() {
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

    #[cfg(all(feature = "wgpu", not(feature = "tch-gpu")))]
    {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        train::<Autodiff<Wgpu>>(config, WgpuDevice::default());
    }

    #[cfg(all(feature = "metal", not(feature = "tch-gpu"), not(feature = "wgpu")))]
    {
        use burn::backend::wgpu::{Metal, WgpuDevice};
        train::<Autodiff<Metal>>(config, WgpuDevice::default());
    }
}

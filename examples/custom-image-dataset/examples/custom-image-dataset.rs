#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use burn::backend::Autodiff;
    use burn::optim::momentum::MomentumConfig;
    use burn::optim::SgdConfig;
    use custom_image_dataset::training::{train, TrainingConfig};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        train::<Autodiff<LibTorch>>(
            TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.,
                nesterov: false,
            }))),
            device,
        );
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::backend::Autodiff;
    use burn::optim::momentum::MomentumConfig;
    use burn::optim::SgdConfig;
    use custom_image_dataset::training::{train, TrainingConfig};

    pub fn run() {
        train::<Autodiff<Wgpu>>(
            TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.,
                nesterov: false,
            }))),
            WgpuDevice::default(),
        );
    }
}

fn main() {
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}

use burn::{optim::RmsPropConfig, tensor::backend::AutodiffBackend};

use wgan::{model::ModelConfig, training::TrainingConfig};

pub fn launch<B: AutodiffBackend>(device: B::Device) {
    let config = TrainingConfig::new(
        ModelConfig::new(),
        RmsPropConfig::new()
            .with_alpha(0.99)
            .with_momentum(0.0)
            .with_epsilon(0.00000008)
            .with_weight_decay(None)
            .with_centered(false),
    );

    wgan::training::train::<B>("/tmp/wgan-mnist", config, device);
}

#[cfg(feature = "flex")]
mod flex {
    use burn::backend::{Autodiff, Flex};

    use crate::launch;

    pub fn run() {
        launch::<Autodiff<Flex>>(Default::default());
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };

    use crate::launch;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        launch::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };

    use crate::launch;

    pub fn run() {
        launch::<Autodiff<LibTorch>>(LibTorchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::launch;
    use burn::backend::{Autodiff, wgpu::Wgpu};

    pub fn run() {
        launch::<Autodiff<Wgpu>>(Default::default());
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::launch;
    use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};

    pub fn run() {
        launch::<Autodiff<Cuda>>(CudaDevice::default());
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

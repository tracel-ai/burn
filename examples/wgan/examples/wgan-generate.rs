use burn::tensor::backend::Backend;

pub fn launch<B: Backend>(device: B::Device) {
    wgan::infer::generate::<B>("/tmp/wgan-mnist", device);
}

#[cfg(feature = "flex")]
mod flex {
    use burn::backend::Flex;

    use crate::launch;

    pub fn run() {
        launch::<Flex>(Default::default());
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    use crate::launch;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        launch::<LibTorch>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    use crate::launch;

    pub fn run() {
        launch::<LibTorch>(LibTorchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::launch;
    use burn::backend::wgpu::Wgpu;

    pub fn run() {
        launch::<Wgpu>(Default::default());
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::launch;
    use burn::backend::Cuda;

    pub fn run() {
        launch::<Cuda>(Default::default());
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

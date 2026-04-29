#[cfg(feature = "flex")]
mod flex {
    use burn::backend::{Autodiff, Flex};
    use dqn_agent::training;

    pub fn run() {
        training::run::<Autodiff<Flex>>(Default::default());
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };
    use dqn_agent::training;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        training::run::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
mod wgpu {
    use burn::backend::{
        Autodiff,
        wgpu::{Wgpu, WgpuDevice},
    };
    use dqn_agent::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::backend::{Autodiff, Cuda};
    use dqn_agent::training;

    pub fn run() {
        let device = Default::default();
        training::run::<Autodiff<Cuda>>(device);
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use burn::backend::{Autodiff, Rocm};
    use dqn_agent::training;

    pub fn run() {
        let device = Default::default();
        training::run::<Autodiff<Rocm>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };
    use dqn_agent::training;

    pub fn run() {
        let device = LibTorchDevice::Cpu;
        training::run::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(feature = "remote")]
mod remote {
    use burn::backend::{Autodiff, RemoteBackend};
    use dqn_agent::training;

    pub fn run() {
        training::run::<Autodiff<RemoteBackend>>(Default::default());
    }
}

fn main() {
    #[cfg(feature = "flex")]
    flex::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "remote")]
    remote::run();
}

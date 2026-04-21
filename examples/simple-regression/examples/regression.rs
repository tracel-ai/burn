use burn::{backend::Autodiff, tensor::backend::Backend};
use simple_regression::{inference, training};

static ARTIFACT_DIR: &str = "/tmp/burn-example-regression";

#[cfg(feature = "flex")]
mod flex {
    use burn::backend::flex::{Flex, FlexDevice};

    pub fn run() {
        let device = FlexDevice;
        super::run::<Flex>(device);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        super::run::<LibTorch>(device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::default();
        super::run::<Wgpu>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use simple_regression::training;
    pub fn run() {
        let device = LibTorchDevice::Cpu;
        super::run::<LibTorch>(device);
    }
}

#[cfg(feature = "remote")]
mod remote {
    use burn::backend::{RemoteBackend, remote::RemoteDevice};

    pub fn run() {
        let device = RemoteDevice::default();
        super::run::<RemoteBackend>(device);
    }
}

/// Train a regression model and predict results on a number of samples.
pub fn run<B: Backend>(device: B::Device) {
    training::run::<Autodiff<B>>(ARTIFACT_DIR, device.clone());
    inference::infer::<B>(ARTIFACT_DIR, device)
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
    #[cfg(feature = "remote")]
    remote::run();
}

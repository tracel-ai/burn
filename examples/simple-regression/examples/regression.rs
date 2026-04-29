use burn::tensor::Device;
use simple_regression::{inference, training};

static ARTIFACT_DIR: &str = "/tmp/burn-example-regression";

#[cfg(feature = "flex")]
mod flex {
    use burn::backend::flex::FlexDevice;

    pub fn run() {
        super::run(FlexDevice);
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

        super::run(device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::wgpu::WgpuDevice;

    pub fn run() {
        super::run(WgpuDevice::default());
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::libtorch::LibTorchDevice;
    pub fn run() {
        super::run(LibTorchDevice::Cpu);
    }
}

// #[cfg(feature = "remote")]
// mod remote {
//     use burn::backend::{RemoteBackend, remote::RemoteDevice};

//     pub fn run() {
//         let device = RemoteDevice::default();
//         super::run::<RemoteBackend>(device);
//     }
// }

/// Train a regression model and predict results on a number of samples.
pub fn run(device: impl Into<Device>) {
    let device = device.into();
    training::run(ARTIFACT_DIR, device.clone());
    inference::infer(ARTIFACT_DIR, device)
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

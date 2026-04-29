use burn::tensor::Device;

pub fn launch(device: impl Into<Device>) {
    wgan::infer::generate("/tmp/wgan-mnist", device.into());
}

<<<<<<< HEAD
#[cfg(feature = "ndarray")]
mod ndarray {
    use burn::backend::ndarray::NdArrayDevice;

    pub fn run() {
        crate::launch(NdArrayDevice::Cpu);
=======
#[cfg(feature = "flex")]
mod flex {
    use burn::backend::Flex;

    use crate::launch;

    pub fn run() {
        launch::<Flex>(Default::default());
>>>>>>> main
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
<<<<<<< HEAD
    #[cfg(feature = "ndarray")]
    ndarray::run();
=======
    #[cfg(feature = "flex")]
    flex::run();
>>>>>>> main
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::Autodiff;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        lora_finetuning::run::<Autodiff<NdArray<f32>>>(device);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::Autodiff;
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        lora_finetuning::run::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::Autodiff;
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub fn run() {
        let device = LibTorchDevice::Cpu;
        lora_finetuning::run::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
mod wgpu {
    use burn::backend::Autodiff;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::default();
        lora_finetuning::run::<Autodiff<Wgpu>>(device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::backend::{Autodiff, Cuda};

    pub fn run() {
        let device = Default::default();
        lora_finetuning::run::<Autodiff<Cuda>>(device);
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use burn::backend::{Autodiff, Rocm};

    pub fn run() {
        let device = Default::default();
        lora_finetuning::run::<Autodiff<Rocm>>(device);
    }
}

#[cfg(feature = "remote")]
mod remote {
    use burn::backend::{Autodiff, RemoteBackend};

    pub fn run() {
        lora_finetuning::run::<Autodiff<RemoteBackend>>(Default::default());
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();

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

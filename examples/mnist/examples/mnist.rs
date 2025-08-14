#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::{
        Autodiff,
        ndarray::{NdArray, NdArrayDevice},
    };
    use mnist::training;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        training::run::<Autodiff<NdArray>>(vec![device]);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };
    use mnist::training;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        training::run::<Autodiff<LibTorch>>(vec![device]);
    }
}

#[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
mod wgpu {
    use burn::backend::{
        Autodiff,
        wgpu::{Wgpu, WgpuDevice},
    };
    use mnist::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(vec![device]);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::backend::{
        Autodiff,
        cuda::{Cuda, CudaDevice},
    };
    use mnist::training;

    pub fn run() {
        #[cfg(not(feature = "ddp"))]
        let devices = vec![CudaDevice::default()];
        #[cfg(feature = "ddp")]
        let devices = vec![
            CudaDevice::new(0),
            CudaDevice::new(1),
            CudaDevice::new(2),
            CudaDevice::new(3),
        ];
        training::run::<Autodiff<Cuda>>(devices);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };
    use mnist::training;

    pub fn run() {
        let device = LibTorchDevice::Cpu;
        training::run::<Autodiff<LibTorch>>(vec![device]);
    }
}

#[cfg(feature = "remote")]
mod remote {
    use burn::backend::{Autodiff, RemoteBackend};
    use mnist::training;

    pub fn run() {
        training::run::<Autodiff<RemoteBackend>>(Default::default());
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
    #[cfg(feature = "remote")]
    remote::run();
}

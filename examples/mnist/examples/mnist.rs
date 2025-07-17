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
        training::run::<Autodiff<NdArray>>(vec![device; 4]);
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

#[cfg(any(feature = "wgpu", feature = "metal",))]
mod wgpu {
    use burn::backend::{
        Autodiff,
        wgpu::{Wgpu, WgpuDevice},
    };
    use mnist::training;

    pub fn run() {
        let gpu_device = WgpuDevice::BestAvailable;

        training::run::<Autodiff<Wgpu>>(vec![gpu_device]);
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
        training::run::<Autodiff<RemoteBackend>>(vec![Default::default()]);
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
    #[cfg(any(feature = "wgpu", feature = "metal"))]
    wgpu::run();
    #[cfg(feature = "remote")]
    remote::run();
}

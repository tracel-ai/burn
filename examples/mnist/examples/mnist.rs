#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArrayAutodiffBackend;
    use mnist::training;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        training::run::<NdArrayAutodiffBackend>(device);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::tch::TchDevice;
    use burn::backend::TchAutodiffBackend;
    use mnist::training;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = TchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = TchDevice::Mps;

        training::run::<TchAutodiffBackend>(device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::WgpuAutodiffBackend;
    use mnist::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<WgpuAutodiffBackend>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::tch::TchDevice;
    use burn::backend::TchAutodiffBackend;
    use mnist::training;

    pub fn run() {
        let device = TchDevice::Cpu;
        training::run::<TchAutodiffBackend>(device);
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
    #[cfg(feature = "wgpu")]
    wgpu::run();
}

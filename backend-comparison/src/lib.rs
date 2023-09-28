#[macro_export]
macro_rules! bench_on_backend {
    () => {
        #[cfg(feature = "wgpu")]
        {
            use burn::backend::wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};

            bench::<WgpuBackend<AutoGraphicsApi, f32, i32>>(&WgpuDevice::default());
        }

        #[cfg(feature = "tch-gpu")]
        {
            use burn::backend::{tch::TchDevice, TchBackend};

            #[cfg(not(target_os = "macos"))]
            let device = TchDevice::Cuda(0);
            #[cfg(target_os = "macos")]
            let device = TchDevice::Mps;
            bench::<TchBackend>(&device);
        }

        #[cfg(feature = "tch-cpu")]
        {
            use burn::backend::{tch::TchDevice, TchBackend};

            let device = TchDevice::Cpu;
            bench::<TchBackend>(&device);
        }

        #[cfg(any(
            feature = "ndarray",
            feature = "ndarray-blas-netlib",
            feature = "ndarray-blas-openblas",
            feature = "ndarray-blas-accelerate",
        ))]
        {
            use burn::backend::ndarray::NdArrayDevice;
            use burn::backend::NdArrayBackend;

            let device = NdArrayDevice::Cpu;
            bench::<NdArrayBackend>(&device);
        }
    };
}

#[macro_export]
macro_rules! bench_on_backend {
    () => {
        #[cfg(feature = "wgpu-fusion")]
        {
            use burn::backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};
            use burn::backend::Fusion;

            bench::<Fusion<Wgpu<AutoGraphicsApi, f32, i32>>>(&WgpuDevice::default());
        }

        #[cfg(feature = "wgpu")]
        {
            use burn::backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};

            bench::<Wgpu<AutoGraphicsApi, f32, i32>>(&WgpuDevice::default());
        }

        #[cfg(feature = "tch-gpu")]
        {
            use burn::backend::{libtorch::LibTorchDevice, LibTorch};

            #[cfg(not(target_os = "macos"))]
            let device = LibTorchDevice::Cuda(0);
            #[cfg(target_os = "macos")]
            let device = LibTorchDevice::Mps;
            bench::<LibTorch>(&device);
        }

        #[cfg(feature = "tch-cpu")]
        {
            use burn::backend::{libtorch::LibTorchDevice, LibTorch};

            let device = LibTorchDevice::Cpu;
            bench::<LibTorch>(&device);
        }

        #[cfg(any(
            feature = "ndarray",
            feature = "ndarray-blas-netlib",
            feature = "ndarray-blas-openblas",
            feature = "ndarray-blas-accelerate",
        ))]
        {
            use burn::backend::ndarray::NdArrayDevice;
            use burn::backend::NdArray;

            let device = NdArrayDevice::Cpu;
            bench::<NdArray>(&device);
        }

        #[cfg(feature = "candle-cpu")]
        {
            use burn::backend::candle::CandleDevice;
            use burn::backend::Candle;

            let device = CandleDevice::Cpu;
            bench::<Candle>(&device);
        }

        #[cfg(feature = "candle-cuda")]
        {
            use burn::backend::candle::CandleDevice;
            use burn::backend::Candle;

            let device = CandleDevice::Cuda(0);
            bench::<Candle>(&device);
        }
    };
}

use burn::{
    grad_clipping::GradientClippingConfig, optim::AdamConfig, tensor::backend::AutodiffBackend,
};
use modern_lstm::{model::LstmNetworkConfig, training::TrainingConfig};

pub fn launch<B: AutodiffBackend>(device: B::Device) {
    let config = TrainingConfig::new(
        LstmNetworkConfig::new(),
        // Gradient clipping via optimizer config
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
    );

    modern_lstm::training::train::<B>("/tmp/modern-lstm", config, device);
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    };

    use crate::launch;

    pub fn run() {
        launch::<Autodiff<NdArray>>(NdArrayDevice::Cpu);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::{
        libtorch::{LibTorch, LibTorchDevice},
        Autodiff,
    };

    use crate::launch;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        launch::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        libtorch::{LibTorch, LibTorchDevice},
        Autodiff,
    };

    use crate::launch;

    pub fn run() {
        launch::<Autodiff<LibTorch>>(LibTorchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::launch;
    use burn::backend::{wgpu::Wgpu, Autodiff};

    pub fn run() {
        launch::<Autodiff<Wgpu>>(Default::default());
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::launch;
    use burn::backend::{cuda::CudaDevice, Autodiff, Cuda};

    pub fn run() {
        launch::<Autodiff<Cuda>>(CudaDevice::default());
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
    #[cfg(feature = "cuda")]
    cuda::run();
}

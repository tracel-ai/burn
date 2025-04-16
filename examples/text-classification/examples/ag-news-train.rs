#![recursion_limit = "256"]

use burn::{
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
    tensor::backend::AutodiffBackend,
};

use text_classification::{AgNewsDataset, training::ExperimentConfig};

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch<B: AutodiffBackend>(devices: Vec<B::Device>) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    text_classification::training::train::<B, AgNewsDataset>(
        devices,
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
        "/tmp/text-classification-ag-news",
    );
}

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

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<NdArray<ElemType>>>(vec![NdArrayDevice::Cpu]);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use crate::{ElemType, launch};
    use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        launch::<Autodiff<LibTorch<ElemType>>>(vec![device]);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<LibTorch<ElemType>>>(vec![LibTorchDevice::Cpu]);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, wgpu::Wgpu};

    pub fn run() {
        launch::<Autodiff<Wgpu<ElemType, i32>>>(vec![Default::default()]);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, Vulkan, autodiff::checkpoint::strategy::BalancedCheckpointing};

    pub fn run() {
        type B = Autodiff<Vulkan<ElemType, i32>, BalancedCheckpointing>;
        launch::<B>(vec![Default::default()]);
    }
}

#[cfg(feature = "metal")]
mod metal {
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, Metal};

    pub fn run() {
        launch::<Autodiff<Metal<ElemType, i32>>>(vec![Default::default()]);
    }
}

#[cfg(feature = "remote")]
mod remote {
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, RemoteBackend};

    pub fn run() {
        launch::<Autodiff<RemoteBackend>>(vec![Default::default()]);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, Cuda, autodiff::checkpoint::strategy::BalancedCheckpointing};

    pub fn run() {
        launch::<Autodiff<Cuda<ElemType, i32>, BalancedCheckpointing>>(vec![Default::default()]);
    }
}

#[cfg(feature = "hip")]
mod hip {
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, Hip};

    pub fn run() {
        launch::<Autodiff<Hip<ElemType, i32>>>(vec![Default::default()]);
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
    #[cfg(feature = "hip")]
    hip::run();
    #[cfg(feature = "remote")]
    remote::run();
    #[cfg(feature = "vulkan")]
    vulkan::run();
    #[cfg(feature = "metal")]
    metal::run();
}

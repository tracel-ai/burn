use burn::{
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
    tensor::backend::AutodiffBackend,
};

use text_classification::{DbPediaDataset, training::ExperimentConfig};

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch<B: AutodiffBackend>(device: B::Device) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4).with_norm_first(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    text_classification::training::train::<B, DbPediaDataset>(
        burn::train::ExecutionStrategy::SingleDevice(device),
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-classification-db-pedia",
    );
}

#[cfg(feature = "flex")]
mod flex {
    use crate::launch;
    use burn::backend::{Autodiff, Flex};

    pub fn run() {
        launch::<Autodiff<Flex>>(Default::default());
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };

    use crate::{ElemType, launch};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        launch::<Autodiff<LibTorch<ElemType>>>(device);
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
        launch::<Autodiff<LibTorch<ElemType>>>(LibTorchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::{
        Autodiff,
        wgpu::{Wgpu, WgpuDevice},
    };

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<Wgpu<ElemType, i32>>>(WgpuDevice::default());
    }
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
}

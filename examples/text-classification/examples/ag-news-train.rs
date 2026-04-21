#![recursion_limit = "256"]

#[cfg(feature = "ddp")]
use burn::tensor::backend::distributed::{DistributedBackend, DistributedConfig, ReduceOperation};
use burn::{
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::*,
    tensor::backend::{AutodiffBackend, DeviceId},
    train::ExecutionStrategy,
};

use text_classification::{AgNewsDataset, training::ExperimentConfig};

#[cfg(not(any(feature = "f16", feature = "flex32")))]
#[allow(unused)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;
#[cfg(feature = "flex32")]
type ElemType = burn::tensor::flex32;

#[cfg(not(feature = "ddp"))]
pub fn launch_multi<B: AutodiffBackend>() {
    let type_id = 0;
    let num_devices = B::device_count(type_id);

    let devices = (0..num_devices)
        .map(|i| B::Device::from_id(DeviceId::new(type_id, i as u16)))
        .collect();

    launch::<B>(ExecutionStrategy::MultiDevice(
        devices,
        burn::train::MultiDeviceOptim::OptimSharded,
    ))
}

#[cfg(feature = "ddp")]
pub fn launch_multi<B: AutodiffBackend + DistributedBackend>() {
    let type_id = 0;
    let num_devices = B::device_count(type_id);

    let devices = (0..num_devices)
        .map(|i| B::Device::from_id(DeviceId::new(type_id, i as u16)))
        .collect();

    launch::<B>(ExecutionStrategy::ddp(
        devices,
        DistributedConfig {
            all_reduce_op: ReduceOperation::Mean,
        },
    ))
}

pub fn launch<B: AutodiffBackend>(strategy: ExecutionStrategy<B>) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    text_classification::training::train::<B, AgNewsDataset>(
        strategy,
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
        "/tmp/text-classification-ag-news",
    );
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use super::*;
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

        launch::<Autodiff<LibTorch<ElemType>>>(ExecutionStrategy::SingleDevice(device));
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use super::*;
    use burn::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<LibTorch<ElemType>>>(ExecutionStrategy::SingleDevice(
            LibTorchDevice::Cpu,
        ));
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, Wgpu};

    pub fn run() {
        launch::<Autodiff<Wgpu<ElemType, i32>>>(
            ExecutionStrategy::SingleDevice(Default::default()),
        );
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, Vulkan, autodiff::checkpoint::strategy::BalancedCheckpointing};

    pub fn run() {
        type B = Autodiff<Vulkan<ElemType, i32>, BalancedCheckpointing>;
        launch::<B>(ExecutionStrategy::SingleDevice(Default::default()));
    }
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, Metal};

    pub fn run() {
        launch::<Autodiff<Metal<ElemType, i32>>>(ExecutionStrategy::SingleDevice(
            Default::default(),
        ));
    }
}

#[cfg(feature = "remote")]
mod remote {
    use super::*;
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, RemoteBackend};

    pub fn run() {
        launch::<Autodiff<RemoteBackend>>(ExecutionStrategy::SingleDevice(Default::default()));
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::{ElemType, launch_multi};
    use burn::backend::{Autodiff, Cuda, autodiff::checkpoint::strategy::BalancedCheckpointing};

    pub fn run() {
        launch_multi::<Autodiff<Cuda<ElemType, i32>, BalancedCheckpointing>>();
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use super::*;
    use crate::{ElemType, launch};
    use burn::backend::{Autodiff, Rocm, autodiff::checkpoint::strategy::BalancedCheckpointing};

    pub fn run() {
        launch::<Autodiff<Rocm<ElemType, i32>, BalancedCheckpointing>>(
            ExecutionStrategy::SingleDevice(Default::default()),
        );
    }
}

#[cfg(feature = "flex")]
mod flex {
    use super::*;
    use crate::launch;
    use burn::backend::{Autodiff, Flex, autodiff::checkpoint::strategy::BalancedCheckpointing};

    pub fn run() {
        launch::<Autodiff<Flex, BalancedCheckpointing>>(ExecutionStrategy::SingleDevice(
            Default::default(),
        ));
    }
}

fn main() {
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "remote")]
    remote::run();
    #[cfg(feature = "vulkan")]
    vulkan::run();
    #[cfg(feature = "metal")]
    metal::run();
    #[cfg(feature = "flex")]
    flex::run();
}

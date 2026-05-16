#![recursion_limit = "256"]

#[cfg(feature = "ddp")]
use burn::tensor::backend::distributed::{DistributedBackend, DistributedConfig, ReduceOperation};
use burn::{
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
    tensor::{DType, Device, Element},
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

#[cfg(all(feature = "cuda", not(feature = "ddp")))]
pub fn launch_multi() {
    let devices = Device::enumerate(burn::tensor::DeviceType::Cuda);

    devices
        .iter()
        .for_each(|d| d.set_default_dtypes(ElemType::dtype(), DType::I32).unwrap());

    launch(ExecutionStrategy::MultiDevice(
        devices,
        burn::train::MultiDeviceOptim::OptimSharded,
    ))
}

#[cfg(all(feature = "cuda", feature = "ddp"))]
pub fn launch_multi<B: AutodiffBackend + DistributedBackend>() {
    let devices = Device::enumerate(burn::tensor::DeviceType::Cuda);

    devices
        .iter()
        .for_each(|d| d.set_default_dtypes(ElemType::dtype(), DType::I32).unwrap());

    launch(ExecutionStrategy::ddp(
        devices,
        DistributedConfig {
            all_reduce_op: ReduceOperation::Mean,
        },
    ))
}

pub fn launch_single(device: impl Into<Device>) {
    let mut device = device.into();
    device
        .set_default_dtypes(ElemType::dtype(), DType::I32)
        .unwrap();

    launch(ExecutionStrategy::SingleDevice(device))
}

pub fn launch(strategy: ExecutionStrategy) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    text_classification::training::train::<AgNewsDataset>(
        strategy,
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
        "/tmp/text-classification-ag-news",
    );
}

#[cfg(feature = "flex")]
mod flex {
    use burn::backend::flex::FlexDevice;

    pub fn run() {
        crate::launch_single(FlexDevice);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use burn::backend::libtorch::LibTorchDevice;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        crate::launch_single(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::libtorch::LibTorchDevice;

    pub fn run() {
        crate::launch_single(LibTorchDevice::Cpu);
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
mod wgpu {
    use burn::backend::wgpu::WgpuDevice;

    pub fn run() {
        crate::launch_single(WgpuDevice::default());
    }
}

// #[cfg(feature = "remote")]
// mod remote {
//     use crate::{ElemType, launch};
//     use burn::backend::{Autodiff, RemoteBackend};

//     pub fn run() {
//         launch::<Autodiff<RemoteBackend>>(ExecutionStrategy::SingleDevice(Default::default()));
//     }
// }

#[cfg(feature = "cuda")]
mod cuda {
    pub fn run() {
        crate::launch_multi();
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use super::*;
    use burn::backend::rocm::RocmDevice;

    pub fn run() {
        crate::launch_single(RocmDevice::default());
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
    #[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "remote")]
    remote::run();
    #[cfg(feature = "flex")]
    flex::run();
}

#![recursion_limit = "256"]

// Only the CUDA `launch_multi` path uses these at the top level; the remote module imports
// them locally. Gating on both features avoids an unused-import warning for `remote,ddp`.
#[cfg(all(feature = "ddp", feature = "cuda"))]
use burn::tensor::distributed::{DistributedConfig, ReduceOperation};
use burn::{
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
    tensor::{Device, DeviceConfig, Element},
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
    let mut devices = Device::enumerate(burn::tensor::DeviceType::Cuda);
    devices
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
        .unwrap();

    launch(ExecutionStrategy::MultiDevice(
        devices.into_vec(),
        burn::train::MultiDeviceOptim::OptimSharded,
    ))
}

#[cfg(all(feature = "cuda", feature = "ddp"))]
pub fn launch_multi() {
    let mut devices = Device::enumerate(burn::tensor::DeviceType::Cuda);
    devices
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
        .unwrap();

    launch(ExecutionStrategy::ddp(
        devices.into_vec(),
        DistributedConfig {
            all_reduce_op: ReduceOperation::Mean,
        },
    ))
}

pub fn launch_single(mut device: Device) {
    device
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
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

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::tensor::{Device, DeviceIndex};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = Device::libtorch_cuda(DeviceIndex::Default);
        #[cfg(target_os = "macos")]
        let device = Device::libtorch_mps();

        crate::launch_single(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::tensor::Device;

    pub fn run() {
        crate::launch_single(Device::libtorch());
    }
}

#[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
mod wgpu {
    use burn::tensor::{Device, DeviceKind};

    pub fn run() {
        crate::launch_single(Device::wgpu(DeviceKind::DefaultDevice));
    }
}

#[cfg(feature = "remote")]
mod remote {
    use crate::ElemType;
    #[cfg(feature = "ddp")]
    use burn::tensor::distributed::{DistributedConfig, ReduceOperation};
    use burn::tensor::{Device, DeviceConfig, DeviceType, Element};
    #[cfg(feature = "ddp")]
    use burn::train::ExecutionStrategy;

    /// Address of the `burn-remote` server to train against.
    const ADDRESS: &str = "ws://localhost:3000";

    /// List every device the remote server hosts and train across all of them.
    #[cfg(not(feature = "ddp"))]
    pub fn run() {
        let mut devices = Device::enumerate(DeviceType::remote(ADDRESS));
        devices
            .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
            .unwrap();

        crate::launch_single(devices.into_vec().pop().unwrap());
    }

    /// Same enumeration, but drive the devices with distributed data-parallel training.
    #[cfg(feature = "ddp")]
    pub fn run() {
        let mut devices = Device::enumerate(DeviceType::remote(ADDRESS));
        devices
            .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
            .unwrap();

        crate::launch(ExecutionStrategy::ddp(
            devices.into_vec(),
            DistributedConfig {
                all_reduce_op: ReduceOperation::Mean,
            },
        ));
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    pub fn run() {
        crate::launch_multi();
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use burn::tensor::{Device, DeviceIndex};

    pub fn run() {
        crate::launch_single(Device::rocm(DeviceIndex::Default));
    }
}

#[cfg(feature = "flex")]
mod flex {
    use burn::tensor::Device;

    pub fn run() {
        crate::launch_single(Device::flex());
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

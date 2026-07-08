use burn::{
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
    tensor::{Device, DeviceConfig, Element},
    train::ExecutionStrategy,
};

use text_classification::{DbPediaDataset, training::ExperimentConfig};

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch(strategy: ExecutionStrategy) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4).with_norm_first(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    text_classification::training::train::<DbPediaDataset>(
        strategy,
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-classification-db-pedia",
    );
}

pub fn launch_single(mut device: Device) {
    device
        .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
        .unwrap();

    launch(ExecutionStrategy::SingleDevice(device))
}

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

#[cfg(feature = "flex")]
mod flex {
    use burn::tensor::Device;

    pub fn run() {
        crate::launch_single(Device::flex());
    }
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

#[cfg(feature = "remote-server")]
mod remote {
    #[cfg(feature = "ddp")]
    use burn::tensor::DeviceType;
    #[cfg(feature = "ddp")]
    use burn::tensor::distributed::{DistributedConfig, ReduceOperation};
    #[cfg(feature = "ddp")]
    use burn::train::ExecutionStrategy;
    use burn::{server::RemoteSecret, tensor::Device};

    /// Address of the `burn-remote` server to train against (legacy WebSocket / DDP path only).
    #[cfg(feature = "ddp")]
    const ADDRESS: &str = "ws://localhost:3000";

    /// Derive a stable server identity from a human-friendly topic, so both ends agree on the address
    /// without exchanging keys. The topic acts as a shared secret here (anyone who knows it can host as
    /// this identity), which suits a demo; a real deployment would use `RemoteSecret::random()` and
    /// share its `id()`.
    fn topic_secret(topic: &str) -> RemoteSecret {
        let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
        RemoteSecret::from_bytes(*hash.as_bytes())
    }

    /// Connect to the remote compute server over Iroh and train against its first device.
    ///
    /// Iroh reaches the server by cryptographic identity, not IP:port — it does NAT traversal and
    /// relay fallback for you, so no public IP or port forwarding is required. Both ends derive the
    /// same identity from the shared `topic`.
    #[cfg(not(feature = "ddp"))]
    pub fn run() {
        use iroh::{Endpoint, EndpointId, endpoint::presets};

        let args: Vec<String> = std::env::args().collect();
        let topic = args
            .get(2)
            .map(String::as_str)
            .unwrap_or("db-pedia-train-default");

        let server_id: EndpointId = topic_secret(topic).id();

        println!("topic     : {topic}");
        println!("server id : {server_id}");
        println!("connecting...");

        // A multi-thread runtime is required: `remote_iroh` blocks to establish the session while
        // Iroh drives networking on the runtime's worker threads.
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async move {
            let endpoint = Endpoint::builder(presets::N0)
                .bind()
                .await
                .expect("failed to bind iroh endpoint");
            let device = Device::remote_iroh(&endpoint, server_id, 0);
            println!("connected\n");
            crate::launch_single(device);
        });
    }

    /// Same enumeration, but drive the devices with distributed data-parallel training.
    #[cfg(feature = "ddp")]
    pub fn run() {
        let devices = Device::enumerate(DeviceType::remote(ADDRESS));

        crate::launch_single(ExecutionStrategy::ddp(
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

fn main() {
    #[cfg(feature = "flex")]
    flex::run();
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
    #[cfg(feature = "remote-server")]
    remote::run();
}

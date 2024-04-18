use burn_common::stub::RwLock;
use burn_compute::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{DeallocStrategy, SimpleMemoryManagement, SliceStrategy},
    tune::Tuner,
    ComputeRuntime,
};
use burn_jit::Runtime;
use std::{marker::PhantomData, sync::Arc};

use crate::{
    compiler::CudaCompiler,
    compute::{CudaServer, CudaStorage},
    device::CudaDevice,
    element::{FloatElement, IntElement},
};

#[derive(Debug)]
pub struct CudaRuntime<F: FloatElement, I: IntElement> {
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

static RUNTIME: ComputeRuntime<CudaDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

type Server = CudaServer<SimpleMemoryManagement<CudaStorage>>;

impl<F: FloatElement, I: IntElement> Runtime for CudaRuntime<F, I> {
    type FullPrecisionRuntime = CudaRuntime<f32, i32>;
    type Compiler = CudaCompiler<F, I>;
    type Server = CudaServer<SimpleMemoryManagement<CudaStorage>>;

    type Channel = MutexComputeChannel<CudaServer<SimpleMemoryManagement<CudaStorage>>>;
    type Device = CudaDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            let device = cudarc::driver::CudaDevice::new(device.index).unwrap();
            let storage = CudaStorage::new(device.clone());
            let memory_management =
                SimpleMemoryManagement::new(storage, DeallocStrategy::Never, SliceStrategy::Never);
            let server = CudaServer::new(device, memory_management);

            let tuner_device_id = tuner_device_id();
            ComputeClient::new(
                MutexComputeChannel::new(server),
                Arc::new(RwLock::new(Tuner::new(&tuner_device_id))),
            )
        })
    }

    fn name() -> &'static str {
        "cuda"
    }

    fn require_array_lengths() -> bool {
        true
    }
}

fn tuner_device_id() -> String {
    "cuda".into()
}

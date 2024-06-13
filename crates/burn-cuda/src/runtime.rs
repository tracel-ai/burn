use burn_common::stub::RwLock;
use burn_compute::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::simple::{DeallocStrategy, SimpleMemoryManagement, SliceStrategy},
    tune::Tuner,
    ComputeRuntime,
};
use burn_cube::Runtime;
use std::sync::Arc;

use crate::{
    compiler::CudaCompiler,
    compute::{CudaContext, CudaServer, CudaStorage},
    device::CudaDevice,
};

#[derive(Debug)]
pub struct CudaRuntime;

impl burn_jit::JitRuntime for CudaRuntime {
    type JitDevice = CudaDevice;
    type JitServer = CudaServer<SimpleMemoryManagement<CudaStorage>>;
}

static RUNTIME: ComputeRuntime<CudaDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

type Server = CudaServer<SimpleMemoryManagement<CudaStorage>>;

impl Runtime for CudaRuntime {
    type Compiler = CudaCompiler;
    type Server = CudaServer<SimpleMemoryManagement<CudaStorage>>;

    type Channel = MutexComputeChannel<CudaServer<SimpleMemoryManagement<CudaStorage>>>;
    type Device = CudaDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        fn init(index: usize) -> CudaContext<SimpleMemoryManagement<CudaStorage>> {
            cudarc::driver::result::init().unwrap();
            let device_ptr = cudarc::driver::result::device::get(index as i32).unwrap();

            let ctx = unsafe {
                let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
                cudarc::driver::result::ctx::set_current(ctx).unwrap();
                ctx
            };

            let stream = cudarc::driver::result::stream::create(
                cudarc::driver::result::stream::StreamKind::NonBlocking,
            )
            .unwrap();
            let storage = CudaStorage::new(stream);
            let memory_management = SimpleMemoryManagement::new(
                storage,
                DeallocStrategy::new_period_tick(1),
                SliceStrategy::Ratio(0.8),
            );
            CudaContext::new(memory_management, stream, ctx)
        }

        RUNTIME.client(device, move || {
            let server = CudaServer::new(device.index, Box::new(init));

            let tuner_device_id = tuner_device_id();
            ComputeClient::new(
                MutexComputeChannel::new(server),
                Arc::new(RwLock::new(Tuner::new("cuda", &tuner_device_id))),
            )
        })
    }

    fn name() -> &'static str {
        "cuda"
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn subcube() -> bool {
        true
    }
}

fn tuner_device_id() -> String {
    "cuda".into()
}

use burn_common::stub::RwLock;
use burn_compute::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::simple::{DeallocStrategy, SimpleMemoryManagement, SliceStrategy},
    tune::Tuner,
    ComputeRuntime,
};
use burn_cube::{
    ir::{Elem, FloatKind},
    Feature, FeatureSet, Runtime,
};
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
            let mut server = CudaServer::new(device.index, Box::new(init));
            let mut features = FeatureSet::new(&[Feature::Subcube]);
            let tuner_device_id = tuner_device_id();

            if let Some(wmma_minimum_version) = register_wmma_features(&mut features, &server.archs)
            {
                server.minimum_arch_version =
                    i32::max(server.minimum_arch_version, wmma_minimum_version);
            }

            ComputeClient::new(
                MutexComputeChannel::new(server),
                Arc::new(RwLock::new(Tuner::new("cuda", &tuner_device_id))),
                Arc::new(features),
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

fn register_wmma_features(features: &mut FeatureSet, archs: &[i32]) -> Option<i32> {
    let wmma_minimum_version = 70;
    let mut wmma = false;

    for arch in archs {
        if *arch >= wmma_minimum_version {
            wmma = true;
            break;
        }
    }

    if wmma {
        // Types fully supported.
        for (a, b, c) in [
            (
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F16),
            ),
            (
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F32),
            ),
            (
                Elem::Float(FloatKind::BF16),
                Elem::Float(FloatKind::BF16),
                Elem::Float(FloatKind::F32),
            ),
        ] {
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 16,
                k: 16,
                n: 16,
            });
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 32,
                k: 8,
                n: 16,
            });
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 8,
                k: 32,
                n: 16,
            });
        }
        return Some(wmma_minimum_version);
    }

    None
}
fn tuner_device_id() -> String {
    "cuda".into()
}

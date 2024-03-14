use crate::{
    compiler::wgsl,
    compute::{Adapter, AdapterInfo, WebGPUApi, WgpuServer, WgpuStorage},
    FloatElement, GraphicsApi, IntElement, WgpuDevice,
};
use alloc::sync::Arc;
use burn_common::stub::RwLock;
use burn_compute::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{DeallocStrategy, SimpleMemoryManagement, SliceStrategy},
    tune::Tuner,
};
use burn_jit::Runtime;
use std::marker::PhantomData;

/// Runtime that uses the [wgpu] crate with the wgsl compiler.
///
/// The [graphics api](GraphicsApi), the [float element](FloatElement) and the
/// [int element](IntElement) types are passed as generic.
#[derive(Debug)]
pub struct WgpuRuntime<W: WebGPUApi, G: GraphicsApi, F: FloatElement, I: IntElement> {
    _w: PhantomData<W>,
    _g: PhantomData<G>,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

impl<W: WebGPUApi, G: GraphicsApi, F: FloatElement, I: IntElement> Runtime
    for WgpuRuntime<W, G, F, I>
{
    type FullPrecisionRuntime = WgpuRuntime<W, G, f32, i32>;
    type Compiler = wgsl::WgslCompiler<F, I>;
    type Server = W::Server;
    type Channel = W::Channel;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        W::client::<G>(device)
    }

    fn name() -> &'static str {
        "wgpu"
    }
}

/// The values that control how a WGPU Runtime will perform its calculations.
pub struct RuntimeOptions {
    /// How the buffers are deallocated.
    pub dealloc_strategy: DeallocStrategy,
    /// Control the slicing strategy.
    pub slice_strategy: SliceStrategy,
    /// Control the amount of compute tasks to be aggregated into a single GPU command.
    pub max_tasks: usize,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        let max_tasks = match std::env::var("BURN_WGPU_MAX_TASKS") {
            Ok(value) => value
                .parse::<usize>()
                .expect("BURN_WGPU_MAX_TASKS should be a positive integer."),
            Err(_) => 64, // 64 tasks by default
        };

        Self {
            dealloc_strategy: DeallocStrategy::new_period_tick(max_tasks * 2),
            slice_strategy: SliceStrategy::Ratio(0.8),
            max_tasks,
        }
    }
}

/// Init the client sync, useful to configure the runtime options.
pub fn init_sync<W: WebGPUApi, G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) {
    W::init_sync::<G>(device, options)
}

/// Init the client async, necessary for wasm.
pub async fn init_async<W: WebGPUApi, G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) {
    W::init_async::<G>(device, options).await
}

pub async fn create_client<W: WebGPUApi, G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) -> ComputeClient<
    WgpuServer<W, SimpleMemoryManagement<WgpuStorage<W>>>,
    MutexComputeChannel<WgpuServer<W, SimpleMemoryManagement<WgpuStorage<W>>>>,
> {
    let (device_wgpu, queue, info) = select_device::<W, G>(device).await;

    log::info!(
        "Created wgpu compute server on device {:?} => {:?}",
        device,
        info
    );

    let device = Arc::new(device_wgpu);
    let storage = WgpuStorage::new(device.clone());
    let memory_management =
        SimpleMemoryManagement::new(storage, options.dealloc_strategy, options.slice_strategy);
    let server = WgpuServer::new(memory_management, device, queue, options.max_tasks);
    let channel = MutexComputeChannel::new(server);

    let tuner_device_id = tuner_device_id::<W>(info);
    ComputeClient::new(channel, Arc::new(RwLock::new(Tuner::new(&tuner_device_id))))
}

/// Select the wgpu device and queue based on the provided [device](WgpuDevice).
pub async fn select_device<W: WebGPUApi, G: GraphicsApi>(
    device: &WgpuDevice,
) -> (W::Device, W::Queue, W::AdapterInfo) {
    #[cfg(target_family = "wasm")]
    let adapter = select_adapter::<W, G>(device).await;

    #[cfg(not(target_family = "wasm"))]
    let adapter = select_adapter::<W, G>(device);

    let (device, queue) = W::select_device(&adapter).await;

    (device, queue, adapter.get_info())
}

fn tuner_device_id<W: WebGPUApi>(info: W::AdapterInfo) -> String {
    format!("wgpu-{}-{}", info.device(), info.backend().as_ref())
}

#[cfg(target_family = "wasm")]
async fn select_adapter<W: WebGPUApi, G: GraphicsApi>(_device: &WgpuDevice) -> W::Adapter {
    W::select_adapter::<G>(device)
}

#[cfg(not(target_family = "wasm"))]
fn select_adapter<W: WebGPUApi, G: GraphicsApi>(device: &WgpuDevice) -> W::Adapter {
    W::select_adapter::<G>(device)
}

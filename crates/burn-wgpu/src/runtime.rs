use crate::{
    compiler::wgsl,
    compute::{
        webgpu_select_adapter, webgpu_select_device, WebGPUAdapter, WebGPUAdapterInfo,
        WebGPUDevice, WebGPUQueue, WgpuServer, WgpuStorage,
    },
    FloatElement, GraphicsApi, IntElement, WgpuDevice,
};
use alloc::sync::Arc;
use burn_common::stub::RwLock;
use burn_compute::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{DeallocStrategy, SimpleMemoryManagement, SliceStrategy},
    tune::Tuner,
    ComputeRuntime,
};
use burn_jit::Runtime;
use std::marker::PhantomData;

/// Runtime that uses the [wgpu] crate with the wgsl compiler.
///
/// The [graphics api](GraphicsApi), the [float element](FloatElement) and the
/// [int element](IntElement) types are passed as generic.
pub struct WgpuRuntime<G: GraphicsApi, F: FloatElement, I: IntElement> {
    _g: PhantomData<G>,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

/// The compute instance is shared across all [wgpu runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

type Server = WgpuServer<SimpleMemoryManagement<WgpuStorage>>;

impl<G: GraphicsApi, F: FloatElement, I: IntElement> Runtime for WgpuRuntime<G, F, I> {
    type FullPrecisionRuntime = WgpuRuntime<G, f32, i32>;
    type Compiler = wgsl::WgslCompiler<F, I>;
    type Server = WgpuServer<SimpleMemoryManagement<WgpuStorage>>;

    type Channel = MutexComputeChannel<WgpuServer<SimpleMemoryManagement<WgpuStorage>>>;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            pollster::block_on(create_client::<G>(device))
        })
    }

    fn name() -> &'static str {
        "wgpu"
    }
}

/// Init the client async, necessary for wasm.
pub async fn init_async<G: GraphicsApi>(device: &WgpuDevice) {
    let device = Arc::new(device);
    let client = create_client::<G>(&device).await;

    RUNTIME.register(&device, client)
}

async fn create_client<G: GraphicsApi>(
    device: &WgpuDevice,
) -> ComputeClient<
    WgpuServer<SimpleMemoryManagement<WgpuStorage>>,
    MutexComputeChannel<WgpuServer<SimpleMemoryManagement<WgpuStorage>>>,
> {
    let (device_wgpu, queue, info) = select_device::<G>(device).await;

    log::info!(
        "Created wgpu compute server on device {:?} => {:?}",
        device,
        info
    );

    // TODO: Support a way to modify max_tasks without std.
    let max_tasks = match std::env::var("BURN_WGPU_MAX_TASKS") {
        Ok(value) => value
            .parse::<usize>()
            .expect("BURN_WGPU_MAX_TASKS should be a positive integer."),
        Err(_) => 64, // 64 tasks by default
    };

    let device = Arc::new(device_wgpu);
    let storage = WgpuStorage::new(device.clone());
    let memory_management = SimpleMemoryManagement::new(
        storage,
        DeallocStrategy::new_period_tick(max_tasks * 2),
        SliceStrategy::Ratio(0.8),
    );
    let server = WgpuServer::new(memory_management, device, queue, max_tasks);
    let channel = MutexComputeChannel::new(server);

    let tuner_device_id = tuner_device_id(info);
    ComputeClient::new(channel, Arc::new(RwLock::new(Tuner::new(&tuner_device_id))))
}

/// Select the wgpu device and queue based on the provided [device](WgpuDevice).
pub async fn select_device<G: GraphicsApi>(
    device: &WgpuDevice,
) -> (WebGPUDevice, WebGPUQueue, WebGPUAdapterInfo) {
    #[cfg(target_family = "wasm")]
    let adapter = select_adapter::<G>(device).await;

    #[cfg(not(target_family = "wasm"))]
    let adapter = select_adapter::<G>(device);

    let (device, queue) = webgpu_select_device(&adapter).await;

    (device, queue, adapter.get_info())
}

fn tuner_device_id(info: WebGPUAdapterInfo) -> String {
    format!("wgpu-{}-{}", info.device, info.backend.to_str())
}

#[cfg(target_family = "wasm")]
async fn select_adapter<G: GraphicsApi>(_device: &WgpuDevice) -> WebGPUAdapter {
    webgpu_select_adapter::<G>(device)
}

#[cfg(not(target_family = "wasm"))]
fn select_adapter<G: GraphicsApi>(device: &WgpuDevice) -> WebGPUAdapter {
    webgpu_select_adapter::<G>(device)
}

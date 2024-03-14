use crate::{
    compiler::wgsl,
    compute::{
        webgpu_select_adapter, webgpu_select_device, WebGPUAdapter, WebGPUAdapterInfo,
        WebGPUDevice, WebGPUQueue, WgpuServer, WgpuStorage,
    },
    GraphicsApi, WgpuDevice,
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
use burn_tensor::backend::{DeviceId, DeviceOps};
use std::marker::PhantomData;

/// Runtime that uses the [wgpu] crate with the wgsl compiler.
///
/// The [graphics api](GraphicsApi) type is passed as generic.
#[derive(Debug)]
pub struct WgpuRuntime<G: GraphicsApi> {
    _g: PhantomData<G>,
}

/// The compute instance is shared across all [wgpu runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

type Server = WgpuServer<SimpleMemoryManagement<WgpuStorage>>;

impl<G: GraphicsApi> Runtime for WgpuRuntime<G> {
    type Compiler = wgsl::WgslCompiler;
    type Server = WgpuServer<SimpleMemoryManagement<WgpuStorage>>;

    type Channel = MutexComputeChannel<WgpuServer<SimpleMemoryManagement<WgpuStorage>>>;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            pollster::block_on(create_client::<G>(device, RuntimeOptions::default()))
        })
    }

    fn name() -> &'static str {
        "wgpu"
    }
}

impl DeviceOps for WgpuDevice {
    fn id(&self) -> DeviceId {
        match self {
            WgpuDevice::DiscreteGpu(index) => DeviceId::new(0, *index as u32),
            WgpuDevice::IntegratedGpu(index) => DeviceId::new(1, *index as u32),
            WgpuDevice::VirtualGpu(index) => DeviceId::new(2, *index as u32),
            WgpuDevice::Cpu => DeviceId::new(3, 0),
            WgpuDevice::BestAvailable => DeviceId::new(4, 0),
        }
    }
}

/// The values that control how a WGPU Runtime will perform its calculations.
pub struct RuntimeOptions {
    /// How the buffers are deallocated.
    pub dealloc_strategy: DeallocStrategy,
    /// Control the slicing strategy.
    pub slice_strategy: SliceStrategy,
    /// Control the amount of compute tasks to be aggregated into a single GPU command.
    pub tasks_max: usize,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        const DEFAULT_MAX_TASKS: usize = 16;

        let tasks_max = match std::env::var("BURN_WGPU_MAX_TASKS") {
            Ok(value) => value
                .parse::<usize>()
                .expect("BURN_WGPU_MAX_TASKS should be a positive integer."),
            Err(_) => DEFAULT_MAX_TASKS,
        };

        Self {
            dealloc_strategy: DeallocStrategy::new_period_tick(tasks_max * 2),
            slice_strategy: SliceStrategy::Ratio(0.8),
            tasks_max,
        }
    }
}

/// Init the client sync, useful to configure the runtime options.
pub fn init_sync<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) {
    let device = Arc::new(device);
    let client = pollster::block_on(create_client::<G>(&device, options));

    RUNTIME.register(&device, client)
}

/// Init the client async, necessary for wasm.
pub async fn init_async<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) {
    let device = Arc::new(device);
    let client = create_client::<G>(&device, options).await;

    RUNTIME.register(&device, client)
}

async fn create_client<G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
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

    let device = Arc::new(device_wgpu);
    let storage = WgpuStorage::new(device.clone());
    let memory_management =
        SimpleMemoryManagement::new(storage, options.dealloc_strategy, options.slice_strategy);
    let server = WgpuServer::new(memory_management, device, queue, options.tasks_max);
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

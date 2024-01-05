use super::WgpuServer;
use crate::{compute::WgpuStorage, GraphicsApi, WgpuDevice};
use alloc::sync::Arc;
use burn_compute::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{DeallocStrategy, SimpleMemoryManagement, SliceStrategy},
    tune::Tuner,
    Compute,
};
use spin::Mutex;
use wgpu::DeviceDescriptor;

type MemoryManagement = SimpleMemoryManagement<WgpuStorage>;
/// Wgpu [compute server](WgpuServer)
pub type Server = WgpuServer<MemoryManagement>;
type Channel = MutexComputeChannel<Server>;

/// Wgpu [compute client](ComputeClient) to communicate with the [compute server](WgpuServer).
pub type WgpuComputeClient = ComputeClient<Server, Channel>;
/// Wgpu [server handle](burn_compute::server::Handle).
pub type WgpuHandle = burn_compute::server::Handle<Server>;

/// Compute handle for the wgpu backend.
static COMPUTE: Compute<WgpuDevice, WgpuServer<MemoryManagement>, Channel> = Compute::new();

/// Get the [compute client](ComputeClient) for the given [device](WgpuDevice).
pub fn compute_client<G: GraphicsApi>(device: &WgpuDevice) -> ComputeClient<Server, Channel> {
    println!("dbg 3_1");
    let device = Arc::new(device);
    println!("dbg 3_2");
    COMPUTE.client(&device, move || {
        println!("dbg 5_1");
        pollster::block_on(create_client::<G>(&device))
    })
}

/// Init the client async, necessary for wasm.
pub async fn init_async<G: GraphicsApi>(device: &WgpuDevice) {
    let device = Arc::new(device);
    let client = create_client::<G>(&device).await;

    COMPUTE.register(&device, client)
}

async fn create_client<G: GraphicsApi>(device: &WgpuDevice) -> ComputeClient<Server, Channel> {
    println!("dbg 6_1");
    let (device_wgpu, queue, info) = select_device::<G>(device).await;

    println!("dbg 6_2");
    log::info!(
        "Created wgpu compute server on device {:?} => {:?}",
        device,
        info
    );

    println!("dbg 6_3");
    // TODO: Support a way to modify max_tasks without std.
    let max_tasks = match std::env::var("BURN_WGPU_MAX_TASKS") {
        Ok(value) => value
            .parse::<usize>()
            .expect("BURN_WGPU_MAX_TASKS should be a positive integer."),
        Err(_) => 64, // 64 tasks by default
    };

    println!("dbg 6_4");
    let device = Arc::new(device_wgpu);
    println!("dbg 6_5");
    let storage = WgpuStorage::new(device.clone());
    println!("dbg 6_6");
    let memory_management = SimpleMemoryManagement::new(
        storage,
        DeallocStrategy::new_period_tick(max_tasks * 2),
        SliceStrategy::Ratio(0.8),
    );
    println!("dbg 6_7");
    let server = WgpuServer::new(memory_management, device, queue, max_tasks);
    println!("dbg 6_8");
    let channel = Channel::new(server);

    println!("dbg 6_9");
    ComputeClient::new(channel, Arc::new(Mutex::new(Tuner::new())))
}

/// Select the wgpu device and queue based on the provided [device](WgpuDevice).
pub async fn select_device<G: GraphicsApi>(
    device: &WgpuDevice,
) -> (wgpu::Device, wgpu::Queue, wgpu::AdapterInfo) {
    println!("dbg 7_1");
    #[cfg(target_family = "wasm")]
    let adapter = select_adapter::<G>(device).await;

    println!("dbg 7_2");
    #[cfg(not(target_family = "wasm"))]
    let adapter = select_adapter::<G>(device);

    println!("dbg 7_3");
    let limits = adapter.limits();

    println!("dbg 7_4");
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits,
            },
            None,
        )
        .await
        .map_err(|err| {
            format!(
                "Unable to request the device with the adapter {:?}, err {:?}",
                adapter.get_info(),
                err
            )
        })
        .unwrap();

    dbg!((&device, &queue, adapter.get_info()));
    (device, queue, adapter.get_info())
}

#[cfg(target_family = "wasm")]
async fn select_adapter<G: GraphicsApi>(_device: &WgpuDevice) -> wgpu::Adapter {
    let instance = wgpu::Instance::default();

    instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase::default())
        .await
        .unwrap()
}

#[cfg(not(target_family = "wasm"))]
fn select_adapter<G: GraphicsApi>(device: &WgpuDevice) -> wgpu::Adapter {
    use wgpu::DeviceType;

    println!("dbg 8_1");
    let instance = wgpu::Instance::default();
    println!("dbg 8_1_1");
    let mut adapters_other = Vec::new();
    println!("dbg 8_1_2");
    let mut adapters = Vec::new();

    println!("dbg 8_2");
    instance
        .enumerate_adapters(G::backend().into())
        .for_each(|adapter| {
            println!("dbg 8_3");
            let device_type = adapter.get_info().device_type;

            println!("dbg 8_4");
            if let DeviceType::Other = device_type {
                adapters_other.push(adapter);
                return;
            }

            println!("dbg 8_5");
            let is_same_type = match device {
                WgpuDevice::DiscreteGpu(_) => device_type == DeviceType::DiscreteGpu,
                WgpuDevice::IntegratedGpu(_) => device_type == DeviceType::IntegratedGpu,
                WgpuDevice::VirtualGpu(_) => device_type == DeviceType::VirtualGpu,
                WgpuDevice::Cpu => device_type == DeviceType::Cpu,
                WgpuDevice::BestAvailable => true,
            };

            println!("dbg 8_6");
            dbg!(&adapter);
            if is_same_type {
                adapters.push(adapter);
            }
        });

    println!("dbg 8_7");
    fn select(
        num: usize,
        error: &str,
        mut adapters: Vec<wgpu::Adapter>,
        mut adapters_other: Vec<wgpu::Adapter>,
    ) -> wgpu::Adapter {
        println!("dbg 8_8");
        if adapters.len() <= num {
            println!("dbg 8_9");
            if adapters_other.len() <= num {
                panic!(
                    "{}, adapters {:?}, other adapters {:?}",
                    error,
                    adapters
                        .into_iter()
                        .map(|adapter| adapter.get_info())
                        .collect::<Vec<_>>(),
                    adapters_other
                        .into_iter()
                        .map(|adapter| adapter.get_info())
                        .collect::<Vec<_>>(),
                );
            } else {
                println!("dbg 8_10");
                return adapters_other.remove(num);
            }
        }

        println!("dbg 8_11");
        adapters.remove(num)
    }

    let adapter = match device {
        WgpuDevice::DiscreteGpu(num) => select(
            *num,
            "No Discrete GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::IntegratedGpu(num) => select(
            *num,
            "No Integrated GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::VirtualGpu(num) => select(
            *num,
            "No Virtual GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::Cpu => select(0, "No CPU device found", adapters, adapters_other),
        WgpuDevice::BestAvailable => {
            println!("dbg 8_12");
            let mut most_performant_adapter = None;
            let mut current_score = -1;

            println!("dbg 8_13");
            adapters
                .into_iter()
                .chain(adapters_other)
                .for_each(|adapter| {
                    println!("dbg 8_14");
                    let info = adapter.get_info();
                    let score = match info.device_type {
                        DeviceType::DiscreteGpu => 5,
                        DeviceType::Other => 4, // Let's be optimistic with the Other device, it's
                        // often a Discrete Gpu.
                        DeviceType::IntegratedGpu => 3,
                        DeviceType::VirtualGpu => 2,
                        DeviceType::Cpu => 1,
                    };
                    println!("dbg 8_15");
                    if score > current_score {
                        most_performant_adapter = Some(adapter);
                        current_score = score;
                    }
                });

            if let Some(adapter) = most_performant_adapter {
                println!("dbg 8_16");
                dbg!(&adapter);
                adapter
            } else {
                panic!("No adapter found for graphics API {:?}", G::default());
            }
        }
    };

    log::info!("Using adapter {:?}", adapter.get_info());

    adapter
}

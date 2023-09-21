use super::WgpuServer;
use crate::{compute::WgpuStorage, context::select_device, GraphicsApi, WgpuDevice};
use alloc::sync::Arc;
use burn_compute::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{DeallocStrategy, SimpleMemoryManagement},
    Compute,
};

type WgpuChannel = MutexComputeChannel<WgpuServer>;

/// Compute handle for the wgpu backend.
static COMPUTE: Compute<WgpuDevice, WgpuServer, WgpuChannel> = Compute::new();

/// Get the [compute client](ComputeClient) for the given [device](WgpuDevice).
pub fn compute_client<G: GraphicsApi>(
    device: &WgpuDevice,
) -> ComputeClient<WgpuServer, WgpuChannel> {
    let device = Arc::new(device);

    COMPUTE.client(&device, move || {
        let (device_wgpu, queue, info) = pollster::block_on(select_device::<G>(&device));

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
            Err(_) => 16, // 16 tasks by default
        };

        let device = Arc::new(device_wgpu);
        let storage = WgpuStorage::new(device.clone());
        // Maximum reusability.
        let memory_management = SimpleMemoryManagement::new(storage, DeallocStrategy::Never);
        let server = WgpuServer::new(memory_management, device, queue, max_tasks);
        let channel = WgpuChannel::new(server);

        ComputeClient::new(channel)
    })
}

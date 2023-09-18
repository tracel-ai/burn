use super::{Kernel, WgpuServer};
use crate::{
    compute::WgpuStorage,
    context::{select_device, WorkGroup},
    kernel::{DynamicKernel, SourceTemplate, StaticKernel},
    GraphicsApi, WgpuDevice,
};
use burn_compute::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{DeallocStrategy, SimpleMemoryManagement},
    Compute,
};
use std::{marker::PhantomData, sync::Arc};

type WgpuChannel = MutexComputeChannel<WgpuServer>;

/// Compute handle for the wgpu backend.
static COMPUTE: Compute<WgpuDevice, WgpuServer, WgpuChannel> = Compute::new();

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

pub struct DynamicComputeKernel<K> {
    kernel: K,
    workgroup: WorkGroup,
}

impl<K> Kernel for DynamicComputeKernel<K>
where
    K: DynamicKernel + 'static,
{
    fn source_template(self: Box<Self>) -> SourceTemplate {
        self.kernel.source_template()
    }

    fn id(&self) -> String {
        self.kernel.id()
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}

#[derive(new)]
pub struct StaticComputeKernel<K> {
    workgroup: WorkGroup,
    _kernel: PhantomData<K>,
}

impl<K> Kernel for StaticComputeKernel<K>
where
    K: StaticKernel + 'static,
{
    fn source_template(self: Box<Self>) -> SourceTemplate {
        K::source_template()
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<K>())
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{binary_elemwise, kernel::KernelSettings, AutoGraphicsApi};

    #[test]
    fn can_run_kernel() {
        binary_elemwise!(Add, "+");

        let client = compute_client::<AutoGraphicsApi>(&WgpuDevice::default());

        let lhs: Vec<f32> = vec![0., 1., 2., 3., 4., 5., 6., 7.];
        let rhs: Vec<f32> = vec![10., 11., 12., 6., 7., 3., 1., 0.];
        let info: Vec<u32> = vec![1, 1, 1, 1, 8, 8, 8];

        let lhs = client.create(bytemuck::cast_slice(&lhs));
        let rhs = client.create(bytemuck::cast_slice(&rhs));
        let out = client.empty(core::mem::size_of::<f32>() * 8);
        let info = client.create(bytemuck::cast_slice(&info));

        type Kernel = KernelSettings<Add, f32, i32, 16, 16, 1>;
        let kernel = Box::new(StaticComputeKernel::<Kernel>::new(WorkGroup::new(1, 1, 1)));

        client.execute(kernel, &[&lhs, &rhs, &out, &info]);

        let data = client.read(&out);
        let output: &[f32] = bytemuck::cast_slice(&data);

        assert_eq!(output, [10., 12., 14., 9., 11., 8., 7., 7.]);
    }
}

use burn_backend::{
    DeviceId, TensorMetadata,
    cubecl::dtype_to_elem_type,
    distributed::{CollectiveTensor, DistributedBackend, ReduceOperation},
    tensor::{Device, FloatTensor},
};
use burn_ir::BackendIr;

use crate::{
    CubeBackend, CubeRuntime,
    ops::numeric::{self, zeros_client},
};

/// Reduce a float tensor across the given devices, returning the resolved output.
///
/// Shared by the [`BackendIr`] implementations so the distributed interpreter path (e.g. the
/// remote backend) can drive collective operations.
pub(crate) fn float_all_reduce<B>(
    tensor: FloatTensor<B>,
    op: ReduceOperation,
    device_ids: Vec<DeviceId>,
) -> FloatTensor<B>
where
    B: BackendIr + DistributedBackend,
{
    let output = B::all_reduce(tensor, op, device_ids);
    // Safety: the collective tensor is immediately registered and not accessed before the
    // collective operation resolves.
    unsafe { output.assume_resolved() }
}

/// Resolve the pending collective operations on the given device.
///
/// Shared by the [`BackendIr`] implementations so the distributed interpreter path (e.g. the
/// remote backend) can resolve collective operations.
pub(crate) fn sync_distributed<B>(device: &Device<B>)
where
    B: BackendIr + DistributedBackend,
{
    B::sync_collective(device);
}

impl<R: CubeRuntime> DistributedBackend for CubeBackend<R> {
    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        let device = &tensor.device.clone();
        let out_tensor = if tensor.handle.can_mut() && tensor.is_contiguous() {
            tensor
        } else {
            let zeros_tensor = zeros_client::<R>(
                tensor.client.clone(),
                device.clone(),
                tensor.shape(),
                tensor.dtype(),
            );
            numeric::add(zeros_tensor, tensor)
        };

        let op = match op {
            ReduceOperation::Sum => cubecl::server::ReduceOperation::Sum,
            ReduceOperation::Mean => cubecl::server::ReduceOperation::Mean,
        };

        let mut client = R::client(device);
        client.all_reduce(
            out_tensor.handle.clone(),
            out_tensor.handle.clone(),
            dtype_to_elem_type(out_tensor.dtype),
            device_ids.clone(),
            op,
        );
        CollectiveTensor::new(out_tensor)
    }

    fn sync_collective(device: &Device<Self>) {
        let client = R::client(device);
        client.sync_collective();
    }
}

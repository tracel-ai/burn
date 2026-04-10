use burn_backend::{
    DeviceId, StreamId, TensorMetadata,
    distributed::{DistributedBackend, ReduceOperation},
    tensor::{Device, FloatTensor},
};

use crate::{
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
    ops::numeric::{self, zeros_client},
};

impl<R, F, I, BT> DistributedBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    unsafe fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> FloatTensor<Self> {
        StreamId::executes(tensor.handle.stream, || {
            let device = &tensor.device.clone();
            let out_tensor = if tensor.handle.can_mut() && tensor.is_contiguous() {
                tensor
            } else {
                let out_tensor = zeros_client::<R>(
                    tensor.client.clone(),
                    device.clone(),
                    tensor.shape(),
                    tensor.dtype(),
                );
                numeric::add(out_tensor, tensor)
            };

            let op = match op {
                ReduceOperation::Sum => cubecl::server::ReduceOperation::Sum,
                ReduceOperation::Mean => cubecl::server::ReduceOperation::Mean,
            };

            let client = R::client(device);
            client.all_reduce(
                out_tensor.handle.clone(),
                out_tensor.handle.clone(),
                out_tensor.dtype.into(),
                device_ids.clone(),
                op,
            );
            out_tensor
        })
    }

    fn sync_collective(device: &Device<Self>) {
        let client = R::client(device);
        client.sync_collective();
    }
}

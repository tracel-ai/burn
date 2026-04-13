use burn_backend::{
    DeviceId, DeviceOps, StreamId, TensorMetadata,
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
        // TODO: Test if `StreamId::executes` is always needed.
        // Output tensor must be on the same stream as the original tensor.
        let device = &tensor.device.clone();
        let out_tensor = if tensor.handle.can_mut() && tensor.is_contiguous() {
            tensor
        } else {
            StreamId::executes(tensor.handle.stream, || {
                let zeros_tensor = zeros_client::<R>(
                    tensor.client.clone(),
                    device.clone(),
                    tensor.shape(),
                    tensor.dtype(),
                );
                numeric::add(zeros_tensor, tensor)
            })
        };

        let op = match op {
            ReduceOperation::Sum => cubecl::server::ReduceOperation::Sum,
            ReduceOperation::Mean => cubecl::server::ReduceOperation::Mean,
        };

        let client = R::client(device);

        // println!("cube all_reduce: {:?}", device.id());

        client.all_reduce(
            out_tensor.handle.clone(),
            out_tensor.handle.clone(),
            out_tensor.dtype.into(),
            device_ids.clone(),
            op,
        );
        out_tensor
    }

    fn sync_collective(device: &Device<Self>) {
        let client = R::client(device);

        // println!("cube sync_collective: {:?}", device.id());

        client.sync_collective();
    }
}

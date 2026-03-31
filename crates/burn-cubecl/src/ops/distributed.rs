use burn_backend::{
    DeviceOps,
    distributed::{DistributedBackend, ReduceOperation, TensorRef},
    tensor::Device,
};

use crate::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

impl<R, F, I, BT> DistributedBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    unsafe fn all_reduce_in_place(tensors: Vec<TensorRef<Self>>, op: ReduceOperation) {
        let tensors = tensors.iter().map(|t| unsafe { &*t.0 }).collect::<Vec<_>>();
        let all_ids = tensors.iter().map(|t| t.device.id()).collect::<Vec<_>>();

        for tensor in tensors {
            let device = &tensor.device;
            let client = R::client(device);

            let op = match op {
                ReduceOperation::Sum => cubecl::server::ReduceOperation::Sum,
                ReduceOperation::Mean => cubecl::server::ReduceOperation::Mean,
            };

            client.all_reduce(
                tensor.handle.clone(),
                tensor.handle.clone(),
                tensor.dtype.into(),
                all_ids.clone(),
                op,
            );
        }
    }

    fn sync_collective(device: &Device<Self>) {
        let client = R::client(device);
        client.sync_collective();
    }
}

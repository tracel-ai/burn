use burn_backend::DeviceOps;
use burn_backend::ops::TensorRef;
use burn_backend::{ReduceOperation, ops::CommunicationTensorOps, tensor::Device};

use crate::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement, kernel};

impl<R, F, I, BT> CommunicationTensorOps<Self> for CubeBackend<R, F, I, BT>
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
            let tensor = kernel::into_contiguous_aligned(tensor.clone());
            let client = R::client(device);

            client.all_reduce(
                tensor.handle.clone(),
                tensor.handle.clone(),
                tensor.dtype.into(),
                all_ids.clone(),
                op.into(),
            );
        }
    }

    fn sync_collective(device: &Device<Self>) {
        println!("Definitely cubecl cuda");
        let client = R::client(device);
        client.sync_collective();
    }
}

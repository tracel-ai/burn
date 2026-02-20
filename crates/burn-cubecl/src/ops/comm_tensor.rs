use burn_backend::ops::FloatTensorOps;
use burn_backend::{
    AllReduceStrategy, Backend, ReduceOperation,
    ops::CommunicationTensorOps,
    tensor::{CommunicationTensor, Device, FloatTensor},
};

use crate::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

pub(crate) fn reduce_sum_centralized<B: Backend>(
    tensors: &Vec<B::CommunicationTensorPrimitive>,
    central_device: &B::Device,
) -> B::FloatTensorPrimitive {
    let mut central_tensor = B::float_data_from_comm(&tensors.get(0).unwrap());
    for tensor in tensors {
        let rhs = B::float_to_device(B::float_data_from_comm(tensor), &central_device);
        central_tensor = B::float_add(central_tensor, rhs);
    }

    central_tensor
}

pub(crate) fn all_reduce_inplace_sum_centralized<B: Backend>(
    tensors: Vec<B::CommunicationTensorPrimitive>,
    op: ReduceOperation,
) {
    // Get corresponding devices for each tensor
    let devices: Vec<B::Device> = tensors
        .iter()
        .map(|tensor| B::comm_device(tensor))
        .collect();
    let central_device = devices.get(0).unwrap();

    // Reduce to central device
    // TODO: inplace?
    let mut central_tensor = reduce_sum_centralized::<B>(&tensors, &central_device);

    if op == ReduceOperation::Mean {
        // Apply mean division
        let div = (tensors.len() as f32).into();
        central_tensor = B::float_div_scalar(central_tensor, div);
    }

    // Broadcast result to all
    B::all_broadcast_inplace(central_tensor, tensors);
}

impl<R, F, I, BT> CommunicationTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn all_reduce_inplace(
        tensors: Vec<CommunicationTensor<Self>>,
        strategy: AllReduceStrategy,
        op: ReduceOperation,
    ) {
        match strategy {
            AllReduceStrategy::Centralized => {
                all_reduce_inplace_sum_centralized::<Self>(tensors, op)
            }
            // AllReduceStrategy::Tree(arity) => all_reduce_sum_tree::<B>(tensors, *arity),
            // AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors),
            AllReduceStrategy::Tree(arity) => todo!(),
            AllReduceStrategy::Ring => todo!(),
        };
    }

    // TODO: broadcast should broadcast to all devices even if tensor is not is dest???
    fn all_broadcast_inplace(
        src_tensor: FloatTensor<Self>,
        dest_tensors: Vec<CommunicationTensor<Self>>,
    ) {
        // TODO: highly unoptimized and unsafe
        unsafe {
            // Centralized
            for dest in dest_tensors {
                let device = Self::comm_device(&dest);
                let tensor_float = Self::float_to_device(src_tensor.clone(), &device);
                (**dest.0) = tensor_float;
            }
        }
    }

    fn comm_device(tensor: &CommunicationTensor<Self>) -> Device<Self> {
        unsafe { (**tensor.0).device.clone() }
    }

    fn float_data_from_comm(tensor: &CommunicationTensor<Self>) -> FloatTensor<Self> {
        unsafe { (**tensor.0).clone() }
    }
}

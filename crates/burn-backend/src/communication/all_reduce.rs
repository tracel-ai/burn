use crate::{Backend, ReduceOperation, ops::TensorRef};

pub(crate) fn reduce_sum_centralized<B: Backend>(
    tensors: &Vec<TensorRef<B>>,
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
    tensors: Vec<TensorRef<B>>,
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
    // This way of assigning in-place is very unsafe and inefficient. Native communication ops are always preferred.
    unsafe {
        for dest in tensors {
            let device = B::comm_device(&dest);
            let tensor_float = B::float_to_device(central_tensor.clone(), &device);
            (**dest.0) = tensor_float;
        }
    }
}

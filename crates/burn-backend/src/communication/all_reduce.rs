use crate::{Backend, ReduceOperation, ops::TensorRef};

pub(crate) fn reduce_sum_centralized<B: Backend>(
    mut tensors: Vec<TensorRef<B>>,
    central_device: &B::Device,
) -> B::FloatTensorPrimitive {
    // Safe since tensors shouldn't be accessed other than here at this point.
    let mut central_tensor = unsafe { B::float_from_ref(&tensors.remove(0)) };

    for tensor in tensors {
        let rhs = unsafe { B::float_to_device(B::float_from_ref(&tensor), &central_device) };
        central_tensor = B::float_add(central_tensor, rhs);
    }

    central_tensor
}

// TODO : Tests
pub(crate) fn all_reduce_inplace_centralized<B: Backend>(
    tensors: Vec<TensorRef<B>>,
    op: ReduceOperation,
) {
    // Get corresponding devices for each tensor
    // Safe since tensors shouldn't be accessed other than here at this point
    let devices: Vec<B::Device> = tensors
        .iter()
        .map(|tensor| unsafe { B::comm_device(tensor) })
        .collect();
    let central_device = devices.get(0).unwrap();

    // Reduce to central device
    let mut central_tensor = reduce_sum_centralized::<B>(tensors.clone(), &central_device);

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

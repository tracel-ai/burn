use crate::{
    Backend,
    distributed::{ReduceOperation, TensorRef},
};

pub(crate) fn reduce_sum_centralized<B: Backend>(
    mut tensors: Vec<TensorRef<B>>,
    central_device: &B::Device,
) -> B::FloatTensorPrimitive {
    // Safe since tensors shouldn't be accessed other than here at this point.
    let mut central_tensor = unsafe { (*tensors.remove(0).0).clone() };

    for tensor in tensors {
        let rhs = unsafe { B::float_to_device((*tensor.0).clone(), central_device) };
        central_tensor = B::float_add(central_tensor.clone(), rhs);
    }

    central_tensor
}

// TODO : Tests
pub(crate) unsafe fn all_reduce_inplace_centralized<B: Backend>(
    tensors: Vec<TensorRef<B>>,
    op: ReduceOperation,
) {
    // Get corresponding devices for each tensor
    let devices: Vec<B::Device> = tensors
        .iter()
        .map(|tensor| unsafe { B::float_device(&(*tensor.0)) })
        .collect();
    let central_device = devices.first().unwrap();

    // Reduce to central device
    let mut central_tensor = reduce_sum_centralized::<B>(tensors.clone(), central_device);

    if op == ReduceOperation::Mean {
        // Apply mean division
        let div = (tensors.len() as f32).into();
        central_tensor = B::float_div_scalar(central_tensor, div);
    }

    // Broadcast result to all
    // This way of assigning in-place is very unsafe and inefficient. Native communication ops are always preferred.
    unsafe {
        for dest in tensors {
            let device = B::float_device(&(*dest.0));
            let tensor_float = B::float_to_device(central_tensor.clone(), &device);
            (*dest.0) = tensor_float;
        }
    }
}

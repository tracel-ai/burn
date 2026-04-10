use crate::{Backend, distributed::ReduceOperation, tensor::FloatTensor};

pub(crate) fn reduce_sum_centralized<B: Backend>(
    mut tensors: Vec<FloatTensor<B>>,
    central_device: &B::Device,
) -> B::FloatTensorPrimitive {
    // Safe since tensors shouldn't be accessed other than here at this point.
    let mut central_tensor = tensors.remove(0);

    for tensor in tensors {
        let rhs = B::float_to_device(tensor, central_device);
        central_tensor = B::float_add(central_tensor.clone(), rhs);
    }

    central_tensor
}

// TODO : Tests
pub(crate) fn all_reduce_centralized<B: Backend>(
    tensors: Vec<FloatTensor<B>>,
    op: ReduceOperation,
) -> Vec<FloatTensor<B>> {
    // Get corresponding devices for each tensor
    let devices: Vec<B::Device> = tensors
        .iter()
        .map(|tensor| B::float_device(tensor))
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
    devices
        .iter()
        .map(|d| B::float_to_device(central_tensor.clone(), d))
        .collect()
}

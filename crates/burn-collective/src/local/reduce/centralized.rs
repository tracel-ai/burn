use std::collections::HashMap;

use burn_tensor::backend::Backend;

use crate::PeerId;

/// Sums the tensors on one device and returns the result
#[tracing::instrument(skip(tensors))]
pub(crate) fn reduce_sum_centralized<B: Backend>(
    mut tensors: HashMap<PeerId, B::FloatTensorPrimitive>,
    central: &PeerId,
) -> B::FloatTensorPrimitive {
    let mut central_tensor = tensors
        .remove(central)
        .expect("Source device id is in the map");
    let central_device = B::float_device(&central_tensor);

    for (_, tensor) in tensors {
        let rhs = B::float_to_device(tensor.clone(), &central_device);
        central_tensor = B::float_add(central_tensor, rhs);
    }

    central_tensor
}

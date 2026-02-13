use burn_tensor::backend::{Backend, PeerId};

use crate::local::tensor_map::CollectiveTensorMap;

#[cfg(feature = "tracing")]
use crate::local::tensor_map::get_common_shape;

/// Sums the tensors on one device and returns the result
#[cfg_attr(feature = "tracing", tracing::instrument(
    level="trace",
    skip(tensors),
    fields(shape = ?get_common_shape::<B>(&tensors).unwrap().dims)
))]
pub(crate) fn reduce_sum_centralized<B: Backend>(
    mut tensors: CollectiveTensorMap<B>,
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

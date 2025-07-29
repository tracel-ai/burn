use std::collections::HashMap;

use burn_tensor::backend::Backend;

use crate::PeerId;

/// Sums the tensors on one device and returns the result
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

/// Broadcasts the tensor from one device in a map to all the others
pub(crate) fn broadcast_centralized<B: Backend>(
    mut devices: HashMap<PeerId, B::Device>,
    central: PeerId,
    tensor: B::FloatTensorPrimitive,
) -> HashMap<PeerId, B::FloatTensorPrimitive> {
    let mut output = HashMap::new();

    devices
        .remove(&central)
        .expect("Central device id is in `devices`");
    for (dest, dest_device) in devices {
        let tensor = B::float_to_device(tensor.clone(), &dest_device);
        output.insert(dest, tensor);
    }
    output.insert(central, tensor);

    output
}

/// Perform an all-reduce operation by reducing all tensors on one device, and broadcasting the
/// result to all other devices
///
/// Internally, this is just a call to `reduce` followed by a `broadcast`
pub(crate) fn all_reduce_sum_centralized<B: Backend>(
    tensors: &mut HashMap<PeerId, B::FloatTensorPrimitive>,
) {
    // Get corresponding devices for each peer
    let devices = tensors
        .iter()
        .map(|(id, tensor)| (*id, B::float_device(tensor)))
        .collect::<HashMap<PeerId, B::Device>>();
    let central_device = *tensors.keys().next().unwrap();

    // Reduce to central device
    let central_tensor = reduce_sum_centralized::<B>(core::mem::take(tensors), &central_device);

    // Broadcast result to all
    *tensors = broadcast_centralized::<B>(devices, central_device, central_tensor);
}

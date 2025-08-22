use std::collections::HashMap;

use burn_tensor::backend::Backend;

use crate::PeerId;

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

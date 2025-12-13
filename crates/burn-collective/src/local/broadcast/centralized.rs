use std::collections::HashMap;

use crate::PeerId;
use crate::local::tensor_map::{CollectiveTensorMap, PeerDeviceMap};
use burn_tensor::backend::Backend;

/// Broadcasts the tensor from one device in a map to all the others
#[tracing::instrument(skip(devices, tensor))]
pub(crate) fn broadcast_centralized<B: Backend>(
    mut devices: PeerDeviceMap<B>,
    central: PeerId,
    tensor: B::FloatTensorPrimitive,
) -> CollectiveTensorMap<B> {
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

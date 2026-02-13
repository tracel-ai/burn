use burn_tensor::backend::{Backend, DeviceOps, PeerId};
use std::collections::HashMap;

use crate::local::tensor_map::{CollectiveTensorMap, PeerDeviceMap};

/// Performs a broadcast on the provided tensors in a b-tree structure with `arity`.
///
/// Tensor must be on the device in the `devices` map corresponding to the `root` key.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(devices, tensor))
)]
pub(crate) fn broadcast_tree<B: Backend>(
    mut devices: PeerDeviceMap<B>,
    root: PeerId,
    tensor: B::FloatTensorPrimitive,
    arity: u32,
) -> CollectiveTensorMap<B> {
    // Convert hash map to vector of key-value pairs because order matters
    let mut devices_vec = vec![];
    let root_device = devices.remove(&root).unwrap();
    for (id, tensor) in devices.drain() {
        devices_vec.push((id, tensor));
    }

    // Sort to put devices of the same type together
    devices_vec.sort_by(|a, b| {
        let dev_a = &a.1;
        let dev_b = &b.1;
        dev_a.id().cmp(&dev_b.id())
    });

    // put the root first
    devices_vec.insert(0, (root, root_device));

    // Recursive broadcast
    let out = broadcast_tree_inner::<B>(tensor, devices_vec, arity);

    // put results in a hash map
    let mut tensors = HashMap::new();
    for (id, tensor) in out {
        tensors.insert(id, tensor);
    }

    tensors
}

/// Recursive function that broadcasts tensor across the other devices. Tensor should be on the
/// first device of the list
///
/// Broadcasts the tensor across the devices in the tree in a pre-order traversal.
fn broadcast_tree_inner<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    mut all_devices: Vec<(PeerId, B::Device)>,
    arity: u32,
) -> Vec<(PeerId, B::FloatTensorPrimitive)> {
    let mut parents = vec![];
    let mut children_groups = vec![];

    // Put devices in groups of `arity` + the parent
    while !all_devices.is_empty() {
        let mut children = vec![];
        let parent = all_devices.remove(0);

        for _ in 0..arity {
            if all_devices.is_empty() {
                break;
            }
            children.push(all_devices.remove(0));
        }

        parents.push(parent);
        children_groups.push(children);
    }

    let mut parents = if parents.len() > 1 {
        broadcast_tree_inner::<B>(tensor, parents, arity)
    } else {
        let root = parents.first().unwrap();
        // `tensor` should already be on the root's device, no need to call B::float_to_device
        vec![(root.0, tensor)]
    };

    // Redistribute result from each parent to the respective devices
    let mut tensors = vec![];
    for children in children_groups {
        let parent = parents.remove(0);
        for (child_id, child_device) in children {
            // replace child's tensor with parent's
            let child_tensor = B::float_to_device(parent.1.clone(), &child_device);
            tensors.push((child_id, child_tensor));
        }
        tensors.push(parent);
    }

    tensors
}

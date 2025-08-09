use crate::PeerId;
use burn_tensor::backend::{Backend, DeviceOps};
use std::collections::HashMap;

/// Performs a all-reduce on the provided tensors in a b-tree structure with `arity`.
/// Similar to [reduce_sum_tree](reduce_sum_tree), but this function broadcasts the result with
/// the same tree algorithm.
/// The returned tensors are on the same devices as the corresponding inputs
pub(crate) fn all_reduce_sum_tree<B: Backend>(
    tensors: &mut HashMap<PeerId, B::FloatTensorPrimitive>,
    arity: u32,
) {
    let mut input = vec![];
    for (id, tensor) in tensors.drain() {
        input.push((id, tensor));
    }

    // Sort to put devices of the same type together
    input.sort_by(|a, b| {
        let dev_a = B::float_device(&a.1);
        let dev_b = B::float_device(&b.1);
        dev_a.id().cmp(&dev_b.id())
    });
    // Recursive all-reduce
    let out = all_reduce_sum_tree_inner::<B>(input, arity);

    for (id, tensor) in out {
        tensors.insert(id, tensor);
    }
}

/// Performs a reduce on the provided tensors in a b-tree structure with `arity`.
pub(crate) fn reduce_sum_tree<B: Backend>(
    mut tensors: HashMap<PeerId, B::FloatTensorPrimitive>,
    root: &PeerId,
    arity: u32,
) -> B::FloatTensorPrimitive {
    // Convert hash map to vector of key-value pairs because order matters
    let mut input = vec![];
    let root_tensor = tensors.remove(root).unwrap();
    for (_, tensor) in tensors.drain() {
        input.push(tensor);
    }

    // Sort to put devices of the same type together
    input.sort_by(|a, b| {
        let dev_a = B::float_device(a);
        let dev_b = B::float_device(b);
        dev_a.id().cmp(&dev_b.id())
    });

    // put the root first
    input.insert(0, root_tensor);

    reduce_sum_tree_inner::<B>(input, arity)
}

/// Performs a broadcast on the provided tensors in a b-tree structure with `arity`.
///
/// Tensor must be on the device in the `devices` map corresponding to the `root` key.
pub(crate) fn broadcast_tree<B: Backend>(
    mut devices: HashMap<PeerId, B::Device>,
    root: PeerId,
    tensor: B::FloatTensorPrimitive,
    arity: u32,
) -> HashMap<PeerId, B::FloatTensorPrimitive> {
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

/// Recursive function that sums `tensors` and redistributes the result to the host devices
fn all_reduce_sum_tree_inner<B: Backend>(
    mut tensors: Vec<(PeerId, B::FloatTensorPrimitive)>,
    arity: u32,
) -> Vec<(PeerId, B::FloatTensorPrimitive)> {
    let mut parent_tensors = vec![];
    let mut children_groups = vec![];

    // Phase 1: Sum tensors in groups of `arity` + 1
    while !tensors.is_empty() {
        // Maps ids to devices for each child of this parent
        let mut children = vec![];
        let (parent, mut parent_tensor) = tensors.remove(0);
        let parent_device = B::float_device(&parent_tensor);

        for _ in 0..arity {
            if tensors.is_empty() {
                break;
            }
            let (child, mut child_tensor) = tensors.remove(0);
            let child_device = B::float_device(&child_tensor);
            children.push((child, child_device));
            child_tensor = B::float_to_device(child_tensor, &parent_device);
            parent_tensor = B::float_add(parent_tensor, child_tensor);
        }

        parent_tensors.push((parent, parent_tensor));
        children_groups.push(children);
    }

    if parent_tensors.len() > 1 {
        // Parents are not yet at the root, do the upper part of the tree
        parent_tensors = all_reduce_sum_tree_inner::<B>(parent_tensors, arity);
    }

    // Phase 2: Redistribute result from each parent to the respective devices
    for (parent, parent_tensor) in parent_tensors {
        let children = children_groups.remove(0);
        for (child, child_device) in children {
            // replace child tensors with result
            tensors.push((
                child,
                B::float_to_device(parent_tensor.clone(), &child_device),
            ));
        }
        tensors.push((parent, parent_tensor));
    }

    tensors
}

/// Recursive function that sums `tensors`
///
/// Traverses `tensors` and reduces in a post-order traversal. The first tensor in the list is
/// chosen as the root
fn reduce_sum_tree_inner<B: Backend>(
    mut tensors: Vec<B::FloatTensorPrimitive>,
    arity: u32,
) -> B::FloatTensorPrimitive {
    let mut parents = vec![];
    let mut children_groups = vec![];

    // Sum tensors in groups of `arity` + 1
    while !tensors.is_empty() {
        let mut children = vec![];
        let mut parent_tensor = tensors.remove(0);
        let parent_device = B::float_device(&parent_tensor);

        for _ in 0..arity {
            if tensors.is_empty() {
                break;
            }
            let child_tensor = tensors.remove(0);
            children.push(B::float_device(&child_tensor));
            let rhs = B::float_to_device(child_tensor, &parent_device);
            parent_tensor = B::float_add(parent_tensor, rhs);
        }

        parents.push(parent_tensor);
        children_groups.push(children);
    }

    if parents.len() > 1 {
        // Parents are not yet at the root, do the upper part of the tree
        reduce_sum_tree_inner::<B>(parents, arity)
    } else {
        // Root of tree
        parents.remove(0)
    }
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

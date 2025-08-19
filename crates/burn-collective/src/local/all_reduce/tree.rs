use crate::PeerId;
use burn_tensor::backend::{Backend, DeviceOps};
use std::collections::HashMap;

/// Performs an all-reduce on the provided tensors in a b-tree structure with `arity`.
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

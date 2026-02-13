use crate::local::tensor_map::CollectiveTensorMap;
use burn_tensor::backend::{Backend, DeviceOps, PeerId};

/// Performs a reduce on the provided tensors in a b-tree structure with `arity`.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(tensors))
)]
pub(crate) fn reduce_sum_tree<B: Backend>(
    mut tensors: CollectiveTensorMap<B>,
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

/// Recursive function that sums `tensors`
///
/// Traverses `tensors` and reduces in a post-order traversal. The first tensor in the list is
/// chosen as the root
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(tensors))
)]
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

use burn_tensor::{
    ElementConversion,
    backend::{Backend, DeviceOps},
};

use crate::ReduceKind;

fn sum_tree<B: Backend>(tensors: &mut Vec<B::FloatTensorPrimitive>, arity: u32) {
    if tensors.len() <= 1 {
        // If there's only one tensor, use it directly
        return;
    }

    let mut summed_tensors = vec![];
    let mut children_devices = vec![];

    // Phase 1: Sum tensors in groups of `arity` + 1
    while !tensors.is_empty() {
        let mut devices = vec![];
        let mut parent_tensor = tensors.remove(0);
        let target_device = B::float_device(&parent_tensor);

        for _ in 0..arity {
            if tensors.is_empty() {
                break;
            }
            let tensor = tensors.remove(0);
            devices.push(B::float_device(&tensor));
            let tensor = B::float_to_device(tensor, &target_device);
            parent_tensor = B::float_add(parent_tensor, tensor);
        }

        summed_tensors.push(parent_tensor);
        children_devices.push(devices);
    }

    sum_tree::<B>(&mut summed_tensors, arity);

    // Phase 2: Redistribute result from each parent to the respective devices
    for devices in children_devices {
        let parent_tensor = summed_tensors.remove(0);
        for device in devices {
            // replace child tensors with result
            tensors.push(B::float_to_device(parent_tensor.clone(), &device));
        }
        tensors.push(parent_tensor);
    }
}

pub(crate) fn all_reduce_tree<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    kind: &ReduceKind,
    arity: u32,
) -> Vec<B::FloatTensorPrimitive> {
    // Sort by device id
    tensors.sort_by(|a, b| {
        let dev_a = B::float_device(a).id();
        let dev_b = B::float_device(b).id();

        dev_a.cmp(&dev_b)
    });

    sum_tree::<B>(tensors, arity);

    let mut result: Vec<B::FloatTensorPrimitive> = std::mem::take(tensors);

    let tensor_count = result.len() as f32;
    if *kind == ReduceKind::Mean {
        result = result
            .into_iter()
            .map(|tensor| {
                // Convert to float if necessary
                B::float_div_scalar(tensor, tensor_count.elem())
            })
            .collect();
    }

    result
}

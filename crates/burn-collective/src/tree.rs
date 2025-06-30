use std::cmp;

use burn_tensor::{
    ElementConversion,
    backend::{Backend, DeviceOps},
};

use crate::{ReduceKind, centralized::all_reduce_centralized};

pub(crate) fn sum_tree<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    arity: u32,
) -> B::FloatTensorPrimitive {
    // Sort by device id
    tensors.sort_by(|a, b| {
        let dev_a = B::float_device(a).id();
        let dev_b = B::float_device(b).id();

        dev_a.cmp(&dev_b)
    });

    let tensor_count = tensors.len() as u32;
    if tensor_count > arity {
        // Split tensor vec into chunks
        let chunk_count = cmp::min(arity, tensor_count);
        let chunk_size = tensor_count / chunk_count;
        let chunks: Vec<Vec<B::FloatTensorPrimitive>> = tensors
            .chunks(chunk_size as usize)
            .map(|s| s.into())
            .collect();

        // Recursive reduce
        let mut new_tensors = vec![];
        for mut chunk in chunks {
            new_tensors.push(sum_tree::<B>(&mut chunk, arity));
        }
        all_reduce_centralized::<B>(&mut new_tensors, &ReduceKind::Sum)
    } else {
        all_reduce_centralized::<B>(tensors, &ReduceKind::Sum)
    }
}

pub(crate) fn all_reduce_tree<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    kind: &ReduceKind,
    arity: u32,
) -> B::FloatTensorPrimitive {
    let tensor_count = tensors.len() as f32;
    let mut result = sum_tree::<B>(tensors, arity);

    if *kind == ReduceKind::Mean {
        result = B::float_div_scalar(result, tensor_count.elem());
    }

    result
}

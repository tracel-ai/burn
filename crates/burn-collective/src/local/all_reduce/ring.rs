use super::tree::all_reduce_sum_tree;
use crate::PeerId;
use crate::local::CollectiveTensorMap;
use crate::local::all_reduce::base;
use burn_tensor::{Shape, Slice, TensorMetadata, backend::Backend};
use std::{collections::HashMap, ops::Range};

/// Ring implementation of All-Reduce (Ring-Reduce)
#[tracing::instrument(skip(tensors))]
pub(crate) fn all_reduce_sum_ring<B: Backend>(
    tensors: CollectiveTensorMap<B>,
) -> CollectiveTensorMap<B> {
    // https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model

    // Example: tensors=3, slices=3

    // phase 1
    // o->o  o
    // o  o->oå
    // o  o  o->

    // o  1->o
    // o  o  1->
    // 1->o  o

    // o  1  2
    // 2  o  1
    // 1  2  o

    // phase 2
    // o  1  2->
    // 2->o  1
    // 1  2->o

    // 2->1  2
    // 2  2->1
    // 1  2  2->

    // 2  2  2
    // 2  2  2
    // 2  2  2

    // Verify all shapes are the same
    let shape = base::get_common_shape::<B>(&tensors)
        .expect("Cannot aggregate tensors with different sizes");

    // Chose an axis
    let slice_dim = get_slice_dim(&shape);

    let slice_dim_size = shape[slice_dim];
    let tensor_count = tensors.len();
    if slice_dim_size < tensor_count {
        // Tensor cannot be split into N slices! Use a fallback algorithm: binary tree
        return all_reduce_sum_tree::<B>(tensors, 2);
    }

    // Split tensors into slices
    let mut sliced_tensors = slice_tensors::<B>(tensors, shape, slice_dim);

    // phase 1: aggregate in ring N-1 times (Reduce-Scatter)
    ring_cycles::<B>(&mut sliced_tensors, true);

    // phase 2: share (overwrite) in a ring N-1 times (All-Gather)
    ring_cycles::<B>(&mut sliced_tensors, false);

    // merge slices and put back in result
    sliced_tensors
        .into_iter()
        .map(|(id, slices)| (id, B::float_cat(slices, slice_dim)))
        .collect()
}

/// Get the dimension to slice across: the largest dimension of the shape
pub(crate) fn get_slice_dim(shape: &Shape) -> usize {
    // get dimension with the greatest size.
    shape
        .dims
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

/// With a ring of N tensors, send the tensors N-1 times, either for the first of second phase.
/// During the first phase, the tensor slices are summed.
/// During the second, the slices are replaced.
fn ring_cycles<B: Backend>(
    sliced_tensors: &mut [(PeerId, Vec<B::FloatTensorPrimitive>)],
    is_phase_one: bool,
) {
    let tensor_count = sliced_tensors.len();
    for cycle in 0..(tensor_count - 1) {
        for i in 0..tensor_count {
            let src_tensor_idx = i;
            let dest_tensor_idx = (i + 1) % tensor_count;

            let slice_idx = if is_phase_one {
                (i + (tensor_count - 1) * cycle) % tensor_count
            } else {
                // in phase 2, the starting slice is different (see diagrams)
                (i + 1 + (tensor_count - 1) * cycle) % tensor_count
            };

            let src_slice = sliced_tensors[src_tensor_idx].1.remove(slice_idx);
            let mut dest_slice = sliced_tensors[dest_tensor_idx].1.remove(slice_idx);

            let dest_device = B::float_device(&dest_slice);
            let src_slice_on_dest = B::float_to_device(src_slice.clone(), &dest_device);
            if is_phase_one {
                dest_slice = B::float_add(dest_slice, src_slice_on_dest);
            } else {
                let slices: Vec<Slice> = dest_slice
                    .shape()
                    .dims
                    .iter()
                    .map(|&d| Slice::new(0, Some(d as isize), 1))
                    .collect();

                // in phase 2, we don't sum the two slices, we replace with the new one.
                dest_slice =
                    B::float_slice_assign(dest_slice, slices.as_slice(), src_slice_on_dest);
            }

            sliced_tensors[src_tensor_idx]
                .1
                .insert(slice_idx, src_slice);
            sliced_tensors[dest_tensor_idx]
                .1
                .insert(slice_idx, dest_slice);
        }
    }
}

/// Slice a list of tensors the same way, evenly across a given dimension.
/// The given `shape` should be the same for every tensor.
fn slice_tensors<B: Backend>(
    mut tensors: HashMap<PeerId, B::FloatTensorPrimitive>,
    shape: Shape,
    slice_dim: usize,
) -> Vec<(PeerId, Vec<<B as Backend>::FloatTensorPrimitive>)> {
    // Get slice index ranges
    let ranges = get_ring_reduce_slice_ranges(shape[slice_dim], tensors.len());

    // Slice tensors
    let mut sliced_tensors = vec![];
    for (id, tensor) in tensors.drain() {
        let mut slices = vec![];
        for range in &ranges {
            let full_range = shape
                .dims
                .iter()
                .enumerate()
                .map(|(dim_idx, dim)| {
                    if dim_idx == slice_dim {
                        Slice::from(range.clone())
                    } else {
                        Slice::from(0..*dim)
                    }
                })
                .collect::<Vec<_>>();
            let slice = B::float_slice(tensor.clone(), &full_range);
            slices.push(slice);
        }
        sliced_tensors.push((id, slices));
    }

    sliced_tensors
}

/// Get the index ranges for the slices to split a tensor evently across a given axis.
///
/// * `slice_dim_size` - The size of the dim to slice on
/// * `slice_count` - The number of slices
///
/// Returns a vector of index ranges for each slice.
pub(crate) fn get_ring_reduce_slice_ranges(
    slice_dim_size: usize,
    slice_count: usize,
) -> Vec<Range<usize>> {
    let mut ranges: Vec<Range<usize>> = vec![];

    let slice_size = slice_dim_size.div_ceil(slice_count);

    for i in 0..slice_count {
        let start = i * slice_size;
        let end = start + slice_size;

        ranges.push(Range { start, end });
    }
    ranges.last_mut().unwrap().end = slice_dim_size;

    ranges
}

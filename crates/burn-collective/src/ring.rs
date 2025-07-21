use std::ops::Range;

use burn_tensor::{Shape, TensorMetadata, backend::Backend};

use crate::tree::all_reduce_sum_tree;

/// Ring implementation of All-Reduce (Ring-Reduce)
pub(crate) fn all_reduce_sum_ring<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
) -> Vec<B::FloatTensorPrimitive> {
    // https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model

    // Example: tensors=3, slices=3

    // phase 1
    // o->o  o
    // o  o->o
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
    let shape = get_shape::<B>(tensors).expect("Cannot aggregate tensors with different sizes");

    // Chose an axis
    let slice_dim = get_slice_dim(&shape);

    let dim_size = shape.dims[slice_dim];
    let tensor_count = tensors.len();
    if dim_size < tensor_count {
        // Tensor cannot be split into N slices! Use a fallback algorithm: binary tree
        return all_reduce_sum_tree::<B>(tensors, 2);
    }

    // Split tensors into slices
    let mut sliced_tensors = slice_tensors::<B>(tensors, dim_size, shape, slice_dim);

    // phase 1: aggregate in ring N-1 times (Reduce-Scatter)
    ring_cycles::<B>(&mut sliced_tensors, true);

    // phase 2: share (overwrite) in a ring N-1 times (All-Gather)
    ring_cycles::<B>(&mut sliced_tensors, false);

    // merge slices
    sliced_tensors
        .into_iter()
        .map(|slices| B::float_cat(slices, slice_dim))
        .collect()
}

/// Get the dimension to slice across: the largest dimension of the shape
pub(crate) fn get_slice_dim(shape: &Shape) -> usize {
    // get dimension with greatest size
    shape
        .dims
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

/// Get the shape of the tensors. They should have all the same shape, otherwise None is returned.
fn get_shape<B: Backend>(tensors: &mut Vec<B::FloatTensorPrimitive>) -> Option<Shape> {
    let mut shape = None;

    for tensor in tensors.as_slice() {
        if shape.is_none() {
            shape = Some(tensor.shape());
        } else if tensor.shape() != *shape.as_ref().unwrap() {
            return None;
        }
    }

    shape
}

/// With a ring of N tensors, send the tensors N-1 times, either for the first of second phase.
/// During the first phase, the tensor slices are summed.
/// During the second, the slices are replaced.
fn ring_cycles<B: Backend>(
    sliced_tensors: &mut [Vec<B::FloatTensorPrimitive>],
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

            let src_slice = sliced_tensors[src_tensor_idx].remove(slice_idx);
            let mut dest_slice = sliced_tensors[dest_tensor_idx].remove(slice_idx);

            let dest_device = B::float_device(&dest_slice);
            let src_slice_on_dest = B::float_to_device(src_slice.clone(), &dest_device);
            if is_phase_one {
                dest_slice = B::float_add(dest_slice, src_slice_on_dest);
            } else {
                let ranges: Vec<Range<usize>> = dest_slice
                    .shape()
                    .dims
                    .iter()
                    .map(|d| Range { start: 0, end: *d })
                    .collect();

                // in phase 2, we don't sum the two slices, we replace with the new one.
                dest_slice =
                    B::float_slice_assign(dest_slice, ranges.as_slice(), src_slice_on_dest);
            }

            sliced_tensors[src_tensor_idx].insert(slice_idx, src_slice);
            sliced_tensors[dest_tensor_idx].insert(slice_idx, dest_slice);
        }
    }
}

/// Slice a list of tensors the same way, evenly across a given dimention.
/// The given `shape` should be the same for every tensor.
fn slice_tensors<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    dim_size: usize,
    shape: Shape,
    slice_dim: usize,
) -> Vec<Vec<<B as Backend>::FloatTensorPrimitive>> {
    // Get slice index ranges
    let ranges = get_ranges(dim_size, tensors.len(), shape, slice_dim);

    // Slice tensors
    let mut sliced_tensors = vec![];
    for tensor in tensors.drain(..) {
        let mut slices = vec![];
        for range in &ranges {
            let slice = B::float_slice(tensor.clone(), range);
            slices.push(slice);
        }
        sliced_tensors.push(slices);
    }

    sliced_tensors
}

/// Get the index ranges for the slices to split a tensor evently across a given axis.
/// Returns a vector of dimentions for each slice.
pub(crate) fn get_ranges(
    dim_size: usize,
    tensor_count: usize,
    shape: Shape,
    slice_dim: usize,
) -> Vec<Vec<Range<usize>>> {
    let mut ranges: Vec<Vec<Range<usize>>> = vec![];

    let slice_size = dim_size / tensor_count;
    for i in 0..tensor_count {
        let start = i * slice_size;
        let end = start + slice_size;

        let mut range_vec = vec![];
        for (dim, size) in shape.dims.iter().enumerate() {
            let range = if dim == slice_dim {
                Range { start, end }
            } else {
                Range {
                    start: 0,
                    end: *size,
                }
            };
            range_vec.push(range);
        }

        ranges.push(range_vec);
    }
    ranges.last_mut().unwrap()[slice_dim].end = dim_size;

    ranges
}

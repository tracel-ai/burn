use core::cmp::Ordering;

use crate::{
    Backend, DType, TensorData,
    element::{ElementConversion, ElementOrdered},
    tensor::{BasicOps, IntElem, IntTensor},
};
use alloc::{vec, vec::Vec};
use burn_std::reader::try_read_sync;
use burn_std::{bf16, f16};

/// Macro used to dispatch sort operations based on dtype.
macro_rules! sort_dispatch_dtype {
    ($fn:ident, $data:ident, $($args:expr),*) => {
        match $data.dtype {
            DType::F64 => $fn::<B, f64>($data, $($args),*),
            DType::F32 | DType::Flex32 => $fn::<B, f32>($data, $($args),*),
            DType::F16 => $fn::<B, f16>($data, $($args),*),
            DType::BF16 => $fn::<B, bf16>($data, $($args),*),
            DType::I64 => $fn::<B, i64>($data, $($args),*),
            DType::I32 => $fn::<B, i32>($data, $($args),*),
            DType::I16 => $fn::<B, i16>($data, $($args),*),
            DType::I8 => $fn::<B, i8>($data, $($args),*),
            DType::U64 => $fn::<B, u64>($data, $($args),*),
            DType::U32 => $fn::<B, u32>($data, $($args),*),
            DType::U16 => $fn::<B, u16>($data, $($args),*),
            DType::U8 => $fn::<B, u8>($data, $($args),*),
            DType::Bool | DType::QFloat(_) => unimplemented!("not supported for sorting operations"),
        }
    };
}

/// Sort the elements of the input `tensor` by value along a given dimension.
///
/// This sort is unstable (i.e., may reorder equal elements).
///
/// # Arguments
///
/// * `tensor` - The input tensor.
/// * `dim` - The axis along which to sort.
/// * `descending` - The sorting order.
///
/// # Returns
///
/// A tensor with the same shape as the input tensor, where the elements are sorted by value.
///
/// # Remarks
///
/// This is a fallback solution that used only when the backend doesn't have the corresponding implementation.
/// Ideally, it is supposed to be implemented by the backend and the backend implementation will be resolved
/// by static dispatch. It is not designed for direct usage by users, and not recommended to import
/// or use this function directly.
pub fn sort<B: Backend, K: BasicOps<B>>(
    tensor: K::Primitive,
    dim: usize,
    descending: bool,
) -> K::Primitive {
    let device = K::device(&tensor);
    let msg = "Failed to synchronously read tensor data. This operation is not supported until this backend has a GPU sorting implementation.";
    let data = try_read_sync(K::into_data_async(tensor))
        .expect(msg)
        .expect(msg);

    let data = sort_dispatch_dtype!(sort_data, data, dim, descending);
    K::from_data(data, &device)
}

pub fn sort_data<B: Backend, E: ElementOrdered>(
    mut data: TensorData,
    dim: usize,
    descending: bool,
) -> TensorData {
    let dims = data.shape.clone();
    let data_slice = data.as_mut_slice().unwrap();
    if dims.len() == 1 {
        // 1D sort
        data_slice.sort_unstable_by(|&a, &b| compare(&a, &b, descending));
    } else {
        sort_slice::<B, E>(data_slice, &dims, dim, None, false, descending);
    }

    data
}

/// Sort the elements of the input `tensor` by value along a given dimension.
///
/// This sort is unstable (i.e., may reorder equal elements).
///
/// # Arguments
///
/// * `tensor` - The input tensor.
/// * `dim` - The axis along which to sort.
/// * `descending` - The sorting order.
///
/// # Returns
///
/// A tensor with the same shape as the input tensor and corresponding indices, where
/// the elements are sorted by value and the indices map back to the original input tensor.
///
/// # Remarks
///
/// This is a fallback solution that used only when the backend doesn't have the corresponding implementation.
/// Ideally, it is supposed to be implemented by the backend and the backend implementation will be resolved
/// by static dispatch. It is not designed for direct usage by users, and not recommended to import
/// or use this function directly.
pub fn sort_with_indices<B: Backend, K: BasicOps<B>>(
    tensor: K::Primitive,
    dim: usize,
    descending: bool,
) -> (K::Primitive, IntTensor<B>) {
    let device = K::device(&tensor);
    let msg = "Failed to synchronously read tensor data. This operation is not supported until this backend has a GPU sorting implementation.";
    let data = try_read_sync(K::into_data_async(tensor))
        .expect(msg)
        .expect(msg);

    let (values, indices) = sort_dispatch_dtype!(sort_data_with_indices, data, dim, descending);

    (
        K::from_data(values, &device),
        B::int_from_data(indices, &device),
    )
}

fn sort_data_with_indices<B: Backend, E: ElementOrdered>(
    mut data: TensorData,
    dim: usize,
    descending: bool,
) -> (TensorData, TensorData) {
    let dims = data.shape.clone();
    let mut indices_data = dim_indices::<B>(&dims, dim);
    let data_slice = data.as_mut_slice().unwrap();
    if dims.len() == 1 {
        // 1D sort
        indices_data.sort_unstable_by(|&a, &b| {
            compare(
                &data_slice[a.elem::<i64>() as usize],
                &data_slice[b.elem::<i64>() as usize],
                descending,
            )
        });

        // Permute data in-place by the sorted indices
        let mut indices = indices_data
            .clone()
            .iter()
            .map(|i| i.elem::<i64>() as usize)
            .collect::<Vec<_>>();
        for idx in 0..indices.len() {
            if indices[idx] != idx {
                let mut current_idx = idx;
                loop {
                    let target_idx = indices[current_idx];
                    indices[current_idx] = current_idx;
                    if indices[target_idx] == target_idx {
                        // correct position
                        break;
                    }

                    // Permute data by indices
                    data_slice.swap(current_idx, target_idx);
                    current_idx = target_idx;
                }
            }
        }
    } else {
        sort_slice::<B, E>(
            data_slice,
            &dims,
            dim,
            Some(&mut indices_data),
            true,
            descending,
        );
    }

    (data, TensorData::new(indices_data, dims))
}

/// Returns the indices that sort the elements of the input `tensor` along a given dimension.
///
/// This sort is unstable (i.e., may reorder equal elements).
///
/// # Arguments
///
/// * `tensor` - The input tensor.
/// * `dim` - The axis along which to sort.
/// * `descending` - The sorting order.
///
/// # Returns
///
/// A tensor with the same shape as the input tensor the indices map back to the original input tensor.
///
/// # Remarks
///
/// This is a fallback solution that used only when the backend doesn't have the corresponding implementation.
/// Ideally, it is supposed to be implemented by the backend and the backend implementation will be resolved
/// by static dispatch. It is not designed for direct usage by users, and not recommended to import
/// or use this function directly.
pub fn argsort<B: Backend, K: BasicOps<B>>(
    tensor: K::Primitive,
    dim: usize,
    descending: bool,
) -> IntTensor<B> {
    let device = K::device(&tensor);
    let msg = "Failed to synchronously read tensor data. This operation is not supported until this backend has a GPU sorting implementation.";
    let data = try_read_sync(K::into_data_async(tensor))
        .expect(msg)
        .expect(msg);

    let data = sort_dispatch_dtype!(argsort_data, data, dim, descending);
    B::int_from_data(data, &device)
}

fn argsort_data<B: Backend, E: ElementOrdered>(
    mut data: TensorData,
    dim: usize,
    descending: bool,
) -> TensorData {
    let dims = data.shape.clone();
    let mut indices_data = dim_indices::<B>(&dims, dim);
    if dims.len() == 1 {
        // 1D sort
        let slice = data.as_slice::<E>().unwrap();
        indices_data.sort_unstable_by(|&a, &b| {
            compare(
                &slice[a.elem::<i64>() as usize],
                &slice[b.elem::<i64>() as usize],
                descending,
            )
        });
    } else {
        sort_slice::<B, E>(
            data.as_mut_slice().unwrap(),
            &dims,
            dim,
            Some(&mut indices_data),
            false,
            descending,
        );
    }

    TensorData::new(indices_data, dims)
}

/// Sort the elements by value along a given dimension.
///
/// When `indices` are not provided, the `data` is sorted.
/// Otherwise, the `indices` are sorted based on the value of the elements in `data`,
/// and if `permute_both` is enabled then the data is also sorted.
///
/// This sort is unstable (i.e., may reorder equal elements).
fn sort_slice<B: Backend, E: ElementOrdered>(
    data: &mut [E],
    dims: &[usize],
    dim: usize,
    mut indices: Option<&mut [IntElem<B>]>,
    permute_both: bool,
    descending: bool,
) {
    let ndims = dims.len();
    let strides = compute_strides(dims);
    // Dimensions to access elements to sort
    let mut sort_dims = dims.to_vec();
    sort_dims[dim] = 1;
    let strides_out = compute_strides(&sort_dims);

    // Number of groups to sort
    let num_sorts: usize = dims
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != dim)
        .map(|(_, d)| d)
        .product();

    // TODO: run each sort in parallel
    // run_par!(|| {
    //     iter_range_par!(0, num_sorts).for_each(|id| {...})
    for id in 0..num_sorts {
        let mut index_offset = 0;
        let mut stride_dim = 0;
        let mut shape_dim = 0;
        for d in 0..ndims {
            let stride_input = strides[d];
            let stride_output = strides_out[d];
            let shape_output = sort_dims[d];

            let num_block = id / stride_output % shape_output;

            if d != dim {
                index_offset += num_block * stride_input;
            } else {
                let shape_input = dims[d];
                stride_dim = stride_input;
                shape_dim = shape_input;
                index_offset += num_block;
            }
        }

        // For each group, sort the indices based on the element values
        // NOTE: Sorting methods like `sort_unstable_by` are in-place but we need to sort
        // different views/groups of the underlying data, so the swap is performed on the elements
        // of the (flat index, element value) collection.
        let mut elements = (0..shape_dim)
            .map(|d| {
                let flat_index = d * stride_dim + index_offset;
                let elem = data[flat_index];
                (d, flat_index, elem)
            })
            .collect::<Vec<_>>();

        elements.sort_unstable_by(|&(_, _, a), &(_, _, b)| compare(&a, &b, descending));

        // Permute data in-place by the sorted indices
        for idx in 0..elements.len() {
            if elements[idx].0 != idx {
                let mut current_idx = idx;
                loop {
                    let target_idx = elements[current_idx].0;
                    elements[current_idx].0 = current_idx;
                    if elements[target_idx].0 == target_idx {
                        // correct position
                        break;
                    }

                    if indices.is_none() || permute_both {
                        // Permute data by indices
                        data.swap(elements[current_idx].1, elements[target_idx].1);
                    }

                    if let Some(ref mut indices_data) = indices {
                        // Permute data element indices
                        indices_data.swap(elements[current_idx].1, elements[target_idx].1);
                    }

                    current_idx = target_idx;
                }
            }
        }
    }
}

/// Computes the steps for each dimension when traversing an array.
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    let mut current = 1;

    dims.iter().enumerate().rev().for_each(|(index, val)| {
        strides[index] = current;
        current *= val;
    });

    strides
}

/// Generates the indices for each element along the specified dimension.
fn dim_indices<B: Backend>(dims: &[usize], dim: usize) -> Vec<IntElem<B>> {
    if dims.len() == 1 {
        (0..dims[dim])
            .map(|i| (i as i64).elem::<IntElem<B>>())
            .collect::<Vec<_>>()
    } else {
        // Dimension indices tensor
        let numel_leading_dims: usize = dims[..dim].iter().product();
        let numel_trailing_dims: usize = dims[dim + 1..].iter().product();
        (0..dims[dim])
            .map(|i| [(i as i64).elem::<IntElem<B>>()].repeat(numel_trailing_dims))
            .collect::<Vec<_>>()
            .concat()
            .repeat(numel_leading_dims)
    }
}

/// Compare two elements
fn compare<E: ElementOrdered>(a: &E, b: &E, descending: bool) -> Ordering {
    if descending { b.cmp(a) } else { a.cmp(b) }
}

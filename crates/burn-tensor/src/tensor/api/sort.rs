use core::cmp::Ordering;

use crate::{
    backend::Backend,
    ops::{IntElem, IntTensor},
    BasicOps, Data, Device, Element, ElementComparison, ElementConversion, TensorKind,
};
use alloc::vec::Vec;

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
#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
pub fn sort<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive<D>,
    dim: usize,
    descending: bool,
) -> K::Primitive<D>
where
    <K as BasicOps<B>>::Elem: Element,
{
    let device = K::device(&tensor);
    let data = K::into_data(tensor).read();

    sort_data::<B, D, K>(data, dim, &device, descending)
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
#[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
pub async fn sort<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive<D>,
    dim: usize,
    descending: bool,
) -> K::Primitive<D>
where
    <K as BasicOps<B>>::Elem: Element,
{
    let device = K::device(&tensor);
    let data = K::into_data(tensor).read().await;

    sort_data::<B, D, K>(data, dim, &device, descending)
}

pub fn sort_data<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    mut data: Data<<K as BasicOps<B>>::Elem, D>,
    dim: usize,
    device: &Device<B>,
    descending: bool,
) -> K::Primitive<D>
where
    <K as BasicOps<B>>::Elem: Element,
{
    let dims = data.shape.dims;
    if D == 1 {
        // 1D sort
        data.value
            .sort_unstable_by(|&a, &b| compare(&a, &b, descending));
    } else {
        sort_slice::<B, D, K>(&mut data.value, &dims, dim, None, false, descending);
    }

    K::from_data(data, device)
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
#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
pub fn sort_with_indices<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive<D>,
    dim: usize,
    descending: bool,
) -> (K::Primitive<D>, IntTensor<B, D>)
where
    <K as BasicOps<B>>::Elem: Element,
{
    let device = K::device(&tensor);
    let data = K::into_data(tensor).read();

    sort_data_with_indices::<B, D, K>(data, dim, &device, descending)
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
#[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
pub async fn sort_with_indices<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive<D>,
    dim: usize,
    descending: bool,
) -> (K::Primitive<D>, IntTensor<B, D>)
where
    <K as BasicOps<B>>::Elem: Element,
{
    let device = K::device(&tensor);
    let data = K::into_data(tensor).read().await;

    sort_data_with_indices::<B, D, K>(data, dim, &device, descending)
}

fn sort_data_with_indices<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    mut data: Data<<K as BasicOps<B>>::Elem, D>,
    dim: usize,
    device: &Device<B>,
    descending: bool,
) -> (K::Primitive<D>, IntTensor<B, D>)
where
    <K as BasicOps<B>>::Elem: Element,
{
    let dims = data.shape.dims;
    let mut indices_data = dim_indices::<B, D>(&dims, dim);
    if D == 1 {
        // 1D sort
        indices_data.sort_unstable_by(|&a, &b| {
            compare(
                &data.value[a.elem::<i64>() as usize],
                &data.value[b.elem::<i64>() as usize],
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
                    data.value.swap(current_idx, target_idx);
                    current_idx = target_idx;
                }
            }
        }
    } else {
        sort_slice::<B, D, K>(
            &mut data.value,
            &dims,
            dim,
            Some(&mut indices_data),
            true,
            descending,
        );
    }

    let shape = data.shape.clone();
    (
        K::from_data(data, device),
        B::int_from_data(Data::new(indices_data, shape), device),
    )
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
#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
pub fn argsort<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive<D>,
    dim: usize,
    descending: bool,
) -> IntTensor<B, D>
where
    <K as BasicOps<B>>::Elem: Element,
{
    let device = K::device(&tensor);
    let data = K::into_data(tensor).read();

    argsort_data::<B, D, K>(data, dim, &device, descending)
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
#[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
pub async fn argsort<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive<D>,
    dim: usize,
    descending: bool,
) -> IntTensor<B, D>
where
    <K as BasicOps<B>>::Elem: Element,
{
    let device = K::device(&tensor);
    let data = K::into_data(tensor).read().await;

    argsort_data::<B, D, K>(data, dim, &device, descending)
}

fn argsort_data<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    mut data: Data<<K as BasicOps<B>>::Elem, D>,
    dim: usize,
    device: &Device<B>,
    descending: bool,
) -> IntTensor<B, D>
where
    <K as BasicOps<B>>::Elem: Element,
{
    let dims = data.shape.dims;
    let mut indices_data = dim_indices::<B, D>(&dims, dim);
    if D == 1 {
        // 1D sort
        indices_data.sort_unstable_by(|&a, &b| {
            compare(
                &data.value[a.elem::<i64>() as usize],
                &data.value[b.elem::<i64>() as usize],
                descending,
            )
        });
    } else {
        sort_slice::<B, D, K>(
            &mut data.value,
            &dims,
            dim,
            Some(&mut indices_data),
            false,
            descending,
        );
    }

    B::int_from_data(Data::new(indices_data, data.shape), device)
}

/// Sort the elements by value along a given dimension.
///
/// When `indices` are not provided, the `data` is sorted.
/// Otherwise, the `indices` are sorted based on the value of the elements in `data`,
/// and if `permute_both` is enabled then the data is also sorted.
///
/// This sort is unstable (i.e., may reorder equal elements).
fn sort_slice<B: Backend, const D: usize, K: BasicOps<B>>(
    data: &mut [<K as BasicOps<B>>::Elem],
    dims: &[usize; D],
    dim: usize,
    mut indices: Option<&mut [IntElem<B>]>,
    permute_both: bool,
    descending: bool,
) where
    <K as BasicOps<B>>::Elem: Element,
{
    let strides = compute_strides(dims);
    // Dimensions to access elements to sort
    let mut sort_dims = *dims;
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
        for d in 0..D {
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
fn compute_strides<const D: usize>(dims: &[usize; D]) -> [usize; D] {
    let mut strides = [0; D];
    let mut current = 1;

    dims.iter().enumerate().rev().for_each(|(index, val)| {
        strides[index] = current;
        current *= val;
    });

    strides
}

/// Generates the indices for each element along the specified dimension.
fn dim_indices<B: Backend, const D: usize>(dims: &[usize; D], dim: usize) -> Vec<IntElem<B>> {
    if D == 1 {
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
fn compare<E: ElementComparison>(a: &E, b: &E, descending: bool) -> Ordering {
    if descending {
        b.cmp(a)
    } else {
        a.cmp(b)
    }
}

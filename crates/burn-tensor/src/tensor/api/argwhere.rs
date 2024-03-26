use crate::{
    backend::Backend,
    ops::{BoolTensor, IntTensor},
    Data, Device, ElementConversion, Shape,
};
use alloc::vec::Vec;

/// Compute the indices of the elements that are non-zero, grouped by element.
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// A vector of tensors, one for each dimension of the given tensor, containing the indices of
/// the non-zero elements in that dimension.
///
/// # Remarks
///
/// This is a fallback solution that used only when the backend doesn't have the corresponding implementation.
/// Ideally, it is supposed to be implemented by the backend and the backend implementation will be resolved
/// by static dispatch. It is not designed for direct usage by users, and not recommended to import
/// or use this function directly.
#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
pub fn argwhere<B: Backend, const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, 2> {
    // Size of each output tensor is variable (= number of nonzero elements in the tensor).
    // Reading the data to count the number of truth values might cause sync but is required.
    // let dims = B::bool_shape(&tensor).dims;
    let device = B::bool_device(&tensor);
    let data = B::bool_into_data(tensor).read();

    argwhere_data::<B, D>(data, &device)
}

/// Compute the indices of the elements that are non-zero, grouped by element.
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// A vector of tensors, one for each dimension of the given tensor, containing the indices of
/// the non-zero elements in that dimension.
///
/// # Remarks
///
/// This is a fallback solution that used only when the backend doesn't have the corresponding implementation.
/// Ideally, it is supposed to be implemented by the backend and the backend implementation will be resolved
/// by static dispatch. It is not designed for direct usage by users, and not recommended to import
/// or use this function directly.
#[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
pub async fn argwhere<B: Backend, const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, 2> {
    // Size of each output tensor is variable (= number of nonzero elements in the tensor).
    // Reading the data to count the number of truth values might cause sync but is required.
    let device = B::bool_device(&tensor);
    let data = B::bool_into_data(tensor).read().await;

    argwhere_data::<B, D>(data, &device)
}

fn argwhere_data<B: Backend, const D: usize>(
    data: Data<bool, D>,
    device: &Device<B>,
) -> IntTensor<B, 2> {
    let dims = data.shape.dims;
    let count_nonzero = data.value.iter().filter(|&v| *v).count();

    /// Converts a flat index into a vector of indices for the specified tensor shape
    fn unravel_index<B: Backend, const D: usize>(
        index: usize,
        shape: &[usize; D],
    ) -> Vec<B::IntElem> {
        shape
            .iter()
            .rev()
            .scan(index, |i, size| {
                let dim_idx = *i % size;
                *i /= size;
                Some((dim_idx as i64).elem())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    let indices = data
        .value
        .iter()
        .enumerate()
        .filter_map(|(index, &v)| if v { Some(index) } else { None })
        .map(|index| unravel_index::<B, D>(index, &dims))
        .collect::<Vec<_>>()
        .concat();

    B::int_from_data(Data::new(indices, Shape::new([count_nonzero, D])), device)
}

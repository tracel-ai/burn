//! Repeat a tensor along a dimension.

use alloc::vec::Vec;
use burn_std::Shape;

use crate::tensor::zeroed_bytes;
use crate::{FlexTensor, Layout};

/// Repeat `tensor` along `dim` by `times`.
pub fn repeat_dim(tensor: FlexTensor, dim: usize, times: usize) -> FlexTensor {
    if times == 1 {
        return tensor;
    }

    let ndims = tensor.layout().num_dims();
    assert!(
        dim < ndims,
        "repeat_dim: dim {} out of bounds for tensor with {} dimensions",
        dim,
        ndims
    );

    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let dtype = tensor.dtype();
    let elem_size = crate::tensor::dtype_size(dtype);

    let mut new_dims: Vec<usize> = shape.iter().cloned().collect();
    new_dims[dim] *= times;
    let new_shape = Shape::from(new_dims);

    let src: &[u8] = tensor.bytes();
    let mut dst = zeroed_bytes(new_shape.num_elements(), dtype);

    let inner: usize = shape.iter().skip(dim + 1).product();
    let dim_size = shape[dim];
    let chunk_bytes = dim_size * inner * elem_size;
    let outer: usize = shape.iter().take(dim).product();

    let mut pos = 0;
    for o in 0..outer {
        let start = o * chunk_bytes;
        let end = start + chunk_bytes;
        for _t in 0..times {
            dst[pos..pos + chunk_bytes].copy_from_slice(&src[start..end]);
            pos += chunk_bytes;
        }
    }

    debug_assert_eq!(pos, dst.len());
    FlexTensor::new(dst, Layout::contiguous(new_shape), dtype)
}

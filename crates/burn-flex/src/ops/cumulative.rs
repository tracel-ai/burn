//! Cumulative operations along a dimension.

use alloc::vec;
use burn_backend::Element;
use burn_std::Bytes;
use bytemuck::Pod;
use num_traits::{Bounded, Num};

use crate::{FlexTensor, Layout};

/// Cumulative sum along a dimension.
///
/// For each position along `dim`, output contains the sum of all elements
/// from index 0 up to and including that position.
pub fn cumsum<E: Element + Pod + Default + Copy + Num>(
    tensor: FlexTensor,
    dim: usize,
) -> FlexTensor {
    cumulative_op(tensor, dim, E::zero(), |acc, val| acc + val)
}

/// Cumulative product along a dimension.
pub fn cumprod<E: Element + Pod + Default + Copy + Num>(
    tensor: FlexTensor,
    dim: usize,
) -> FlexTensor {
    cumulative_op(tensor, dim, E::one(), |acc, val| acc * val)
}

/// Generic cumulative operation along a dimension.
///
/// Uses blocked iteration for cache-friendly access: processes contiguous
/// inner blocks together rather than striding across memory one element at
/// a time.
fn cumulative_op<E: Element + Pod + Default + Copy, F>(
    tensor: FlexTensor,
    dim: usize,
    init: E,
    op: F,
) -> FlexTensor
where
    F: Fn(E, E) -> E,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let data: &[E] = tensor.storage();
    let total_size = shape.num_elements();
    let mut result = vec![E::default(); total_size];

    let dim_size = shape[dim];
    // Contiguous block size after the cumulative dimension
    let inner_size: usize = shape[dim + 1..].iter().product();
    // Number of outer blocks (dimensions before the cumulative dimension)
    let outer_size: usize = shape[..dim].iter().product();
    let block_size = dim_size * inner_size;

    if inner_size == 1 {
        // Scalar accumulator path: accumulator stays in a register.
        for outer in 0..outer_size {
            let base = outer * dim_size;
            let mut acc = init;
            for i in 0..dim_size {
                acc = op(acc, data[base + i]);
                result[base + i] = acc;
            }
        }
    } else {
        // Blocked path: process contiguous inner blocks together for
        // cache-friendly access when the cumulative dim is not last.
        let mut acc = vec![init; inner_size];
        for outer in 0..outer_size {
            let base = outer * block_size;
            acc.fill(init);
            for i in 0..dim_size {
                let offset = base + i * inner_size;
                for j in 0..inner_size {
                    acc[j] = op(acc[j], data[offset + j]);
                    result[offset + j] = acc[j];
                }
            }
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(bytes, Layout::contiguous(shape), E::dtype())
}

/// Cumulative operation for half-precision types, accumulating in f32.
fn cumulative_op_half<E: Element + Pod + Default + Copy, F>(
    tensor: FlexTensor,
    dim: usize,
    init: f32,
    op: F,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor
where
    F: Fn(f32, f32) -> f32,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let data: &[E] = tensor.storage();
    let total_size = shape.num_elements();
    let mut result = vec![E::default(); total_size];

    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();
    let outer_size: usize = shape[..dim].iter().product();
    let block_size = dim_size * inner_size;

    // Accumulator buffer for f32 intermediate values
    let mut acc = vec![init; inner_size];

    for outer in 0..outer_size {
        let base = outer * block_size;

        // Reset accumulators
        acc.fill(init);

        for i in 0..dim_size {
            let offset = base + i * inner_size;
            for j in 0..inner_size {
                acc[j] = op(acc[j], to_f32(data[offset + j]));
                result[offset + j] = from_f32(acc[j]);
            }
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(bytes, Layout::contiguous(shape), E::dtype())
}

// Type-specific wrappers

pub fn cumsum_f32(tensor: FlexTensor, dim: usize) -> FlexTensor {
    cumsum::<f32>(tensor, dim)
}

pub fn cumsum_f64(tensor: FlexTensor, dim: usize) -> FlexTensor {
    cumsum::<f64>(tensor, dim)
}

pub fn cumprod_f32(tensor: FlexTensor, dim: usize) -> FlexTensor {
    cumprod::<f32>(tensor, dim)
}

pub fn cumprod_f64(tensor: FlexTensor, dim: usize) -> FlexTensor {
    cumprod::<f64>(tensor, dim)
}

/// Cumulative min along a dimension.
pub fn cummin<E: Element + Pod + Default + Copy + Ord + Bounded>(
    tensor: FlexTensor,
    dim: usize,
) -> FlexTensor {
    cumulative_op(tensor, dim, E::max_value(), Ord::min)
}

pub fn cummin_f32(tensor: FlexTensor, dim: usize) -> FlexTensor {
    cumulative_op(tensor, dim, f32::INFINITY, |acc, val| {
        if val.is_nan() || val < acc { val } else { acc }
    })
}

pub fn cummin_f64(tensor: FlexTensor, dim: usize) -> FlexTensor {
    cumulative_op(tensor, dim, f64::INFINITY, |acc, val| {
        if val.is_nan() || val < acc { val } else { acc }
    })
}

/// Cumulative max along a dimension.
pub fn cummax<E: Element + Pod + Default + Copy + Ord + Bounded>(
    tensor: FlexTensor,
    dim: usize,
) -> FlexTensor {
    cumulative_op(tensor, dim, E::min_value(), Ord::max)
}

pub fn cummax_f32(tensor: FlexTensor, dim: usize) -> FlexTensor {
    cumulative_op(tensor, dim, f32::NEG_INFINITY, |acc, val| {
        if val.is_nan() || val > acc { val } else { acc }
    })
}

pub fn cummax_f64(tensor: FlexTensor, dim: usize) -> FlexTensor {
    cumulative_op(tensor, dim, f64::NEG_INFINITY, |acc, val| {
        if val.is_nan() || val > acc { val } else { acc }
    })
}

pub fn cumsum_half<E: Element + Pod + Default + Copy>(
    tensor: FlexTensor,
    dim: usize,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor {
    cumulative_op_half(tensor, dim, 0.0, |acc, val| acc + val, to_f32, from_f32)
}

pub fn cumprod_half<E: Element + Pod + Default + Copy>(
    tensor: FlexTensor,
    dim: usize,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor {
    cumulative_op_half(tensor, dim, 1.0, |acc, val| acc * val, to_f32, from_f32)
}

pub fn cummin_half<E: Element + Pod + Default + Copy>(
    tensor: FlexTensor,
    dim: usize,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor {
    cumulative_op_half(
        tensor,
        dim,
        f32::INFINITY,
        |acc, val| if val.is_nan() || val < acc { val } else { acc },
        to_f32,
        from_f32,
    )
}

pub fn cummax_half<E: Element + Pod + Default + Copy>(
    tensor: FlexTensor,
    dim: usize,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor {
    cumulative_op_half(
        tensor,
        dim,
        f32::NEG_INFINITY,
        |acc, val| if val.is_nan() || val > acc { val } else { acc },
        to_f32,
        from_f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_cumsum_1d() {
        // [1, 2, 3, 4] -> [1, 3, 6, 10]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let result = cumsum_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumsum_2d_dim0() {
        // [[1, 2], [3, 4], [5, 6]] cumsum along dim 0
        // -> [[1, 2], [4, 6], [9, 12]]
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
        ));
        let result = cumsum_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_cumsum_2d_dim1() {
        // [[1, 2, 3], [4, 5, 6]] cumsum along dim 1
        // -> [[1, 3, 6], [4, 9, 15]]
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        let result = cumsum_f32(tensor, 1);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn test_cumprod_1d() {
        // [1, 2, 3, 4] -> [1, 2, 6, 24]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let result = cumprod_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_cummin_1d() {
        // [3, 1, 4, 1, 5] -> [3, 1, 1, 1, 1]
        let tensor = FlexTensor::from_data(TensorData::new(vec![3.0f32, 1.0, 4.0, 1.0, 5.0], [5]));
        let result = cummin_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cummax_1d() {
        // [3, 1, 4, 1, 5] -> [3, 3, 4, 4, 5]
        let tensor = FlexTensor::from_data(TensorData::new(vec![3.0f32, 1.0, 4.0, 1.0, 5.0], [5]));
        let result = cummax_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 3.0, 4.0, 4.0, 5.0]);
    }

    #[test]
    fn test_cumsum_i64() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![1i64, 2, 3, 4], [4]));
        let result = cumsum::<i64>(tensor, 0);
        let data: Vec<i64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_cummax_2d_dim1() {
        // [[1, 3, 2], [4, 2, 5]] cummax along dim 1
        // -> [[1, 3, 3], [4, 4, 5]]
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0],
            [2, 3],
        ));
        let result = cummax_f32(tensor, 1);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 3.0, 3.0, 4.0, 4.0, 5.0]);
    }

    #[test]
    fn test_cummin_nan_propagation() {
        // [3.0, NaN, 1.0, 2.0] -> [3.0, NaN, NaN, NaN]
        let tensor = FlexTensor::from_data(TensorData::new(vec![3.0f32, f32::NAN, 1.0, 2.0], [4]));
        let result = cummin_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data[0], 3.0);
        assert!(data[1].is_nan());
        assert!(data[2].is_nan());
        assert!(data[3].is_nan());
    }

    #[test]
    fn test_cummax_nan_propagation() {
        // [1.0, NaN, 5.0, 2.0] -> [1.0, NaN, NaN, NaN]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, f32::NAN, 5.0, 2.0], [4]));
        let result = cummax_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data[0], 1.0);
        assert!(data[1].is_nan());
        assert!(data[2].is_nan());
        assert!(data[3].is_nan());
    }

    #[test]
    fn test_cummin_nan_at_start() {
        // [NaN, 1.0, 2.0] -> [NaN, NaN, NaN]
        let tensor = FlexTensor::from_data(TensorData::new(vec![f32::NAN, 1.0, 2.0], [3]));
        let result = cummin_f32(tensor, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!(data[0].is_nan());
        assert!(data[1].is_nan());
        assert!(data[2].is_nan());
    }
}

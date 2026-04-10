//! Mask operations for conditional element replacement.

use alloc::vec::Vec;
use burn_backend::Element;
use burn_std::{Bytes, bf16, f16};

use crate::{FlexTensor, Layout};

/// Allocate a Vec of given length without zeroing.
/// The caller must write every element before reading.
#[cfg(feature = "simd")]
#[inline]
fn uninit_vec<T: Copy>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    #[allow(clippy::uninit_vec)]
    unsafe {
        v.set_len(len);
    }
    v
}

/// Fill tensor elements with a value where mask is true.
///
/// mask_fill(tensor, mask, value) -> tensor with elements replaced where mask is true
pub fn mask_fill<T>(tensor: FlexTensor, mask: FlexTensor, value: T) -> FlexTensor
where
    T: Element + bytemuck::Pod + Copy,
{
    let dtype = tensor.dtype();

    // Broadcast mask to tensor shape if needed
    let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);

    let tensor = tensor.to_contiguous();
    let mask = mask.to_contiguous();

    let shape = tensor.layout().shape().clone();
    let tensor_data: &[T] = tensor.storage();
    let mask_data: &[u8] = mask.bytes();

    let result: Vec<T> = tensor_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&elem, &m)| if m != 0 { value } else { elem })
        .collect();

    FlexTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), dtype)
}

/// Mask fill for f32 (SIMD-accelerated).
pub fn mask_fill_f32(tensor: FlexTensor, mask: FlexTensor, value: f32) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let dtype = tensor.dtype();
        let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);
        let tensor = tensor.to_contiguous();
        let mask = mask.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let len = tensor.storage::<f32>().len();
        let mut out = uninit_vec::<f32>(len);
        crate::simd::mask_fill_f32(tensor.storage(), mask.bytes(), value, &mut out);
        FlexTensor::new(Bytes::from_elems(out), Layout::contiguous(shape), dtype)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_fill(tensor, mask, value)
    }
}

/// Mask fill for f64 (SIMD-accelerated).
pub fn mask_fill_f64(tensor: FlexTensor, mask: FlexTensor, value: f64) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let dtype = tensor.dtype();
        let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);
        let tensor = tensor.to_contiguous();
        let mask = mask.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let len = tensor.storage::<f64>().len();
        let mut out = uninit_vec::<f64>(len);
        crate::simd::mask_fill_f64(tensor.storage(), mask.bytes(), value, &mut out);
        FlexTensor::new(Bytes::from_elems(out), Layout::contiguous(shape), dtype)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_fill(tensor, mask, value)
    }
}

/// Mask fill for f16.
pub fn mask_fill_f16(tensor: FlexTensor, mask: FlexTensor, value: f16) -> FlexTensor {
    mask_fill(tensor, mask, value)
}

/// Mask fill for bf16.
pub fn mask_fill_bf16(tensor: FlexTensor, mask: FlexTensor, value: bf16) -> FlexTensor {
    mask_fill(tensor, mask, value)
}

/// Mask fill for i64 (SIMD-accelerated).
pub fn mask_fill_i64(tensor: FlexTensor, mask: FlexTensor, value: i64) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let dtype = tensor.dtype();
        let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);
        let tensor = tensor.to_contiguous();
        let mask = mask.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let len = tensor.storage::<i64>().len();
        let mut out = uninit_vec::<i64>(len);
        crate::simd::mask_fill_i64(tensor.storage(), mask.bytes(), value, &mut out);
        FlexTensor::new(Bytes::from_elems(out), Layout::contiguous(shape), dtype)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_fill(tensor, mask, value)
    }
}

/// Mask fill for u64.
pub fn mask_fill_u64(tensor: FlexTensor, mask: FlexTensor, value: u64) -> FlexTensor {
    mask_fill(tensor, mask, value)
}

/// Mask fill for bool tensors (SIMD-accelerated).
pub fn mask_fill_bool(tensor: FlexTensor, mask: FlexTensor, value: bool) -> FlexTensor {
    // Preserve the input tensor's bool dtype for the output.
    let out_dtype = burn_std::BoolDType::from(tensor.dtype());
    #[cfg(feature = "simd")]
    {
        let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);
        let tensor = tensor.to_contiguous();
        let mask = mask.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let len = tensor.bytes().len();
        let mut out = uninit_vec::<u8>(len);
        crate::simd::mask_fill_u8(tensor.bytes(), mask.bytes(), value as u8, &mut out);
        crate::ops::comparison::make_bool_tensor(out, shape, out_dtype)
    }
    #[cfg(not(feature = "simd"))]
    {
        let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);
        let tensor = tensor.to_contiguous();
        let mask = mask.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let tensor_data: &[u8] = tensor.bytes();
        let mask_data: &[u8] = mask.bytes();
        let value_u8 = value as u8;
        let result: Vec<u8> = tensor_data
            .iter()
            .zip(mask_data.iter())
            .map(|(&elem, &m)| if m != 0 { value_u8 } else { elem })
            .collect();
        crate::ops::comparison::make_bool_tensor(result, shape, out_dtype)
    }
}

/// Replace elements from value tensor where mask is true.
///
/// mask_where(tensor, mask, value) -> tensor with elements from value where mask is true
pub fn mask_where<T>(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor
where
    T: Element + bytemuck::Pod + Copy,
{
    let dtype = tensor.dtype();
    let (tensor, mask, value) = broadcast_three(tensor, mask, value);

    let shape = tensor.layout().shape().clone();
    let tensor_data: &[T] = tensor.storage();
    let mask_data: &[u8] = mask.bytes();
    let value_data: &[T] = value.storage();

    let result: Vec<T> = tensor_data
        .iter()
        .zip(mask_data.iter())
        .zip(value_data.iter())
        .map(|((&t, &m), &v)| if m != 0 { v } else { t })
        .collect();

    FlexTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), dtype)
}

/// Helper to broadcast three tensors to the same shape.
fn broadcast_three(
    tensor: FlexTensor,
    mask: FlexTensor,
    value: FlexTensor,
) -> (FlexTensor, FlexTensor, FlexTensor) {
    let target_shape =
        crate::ops::expand::broadcast_shape(tensor.layout().shape(), mask.layout().shape());
    let target_shape = crate::ops::expand::broadcast_shape(&target_shape, value.layout().shape());

    let tensor = if tensor.layout().shape() == &target_shape {
        tensor
    } else {
        crate::ops::expand::expand(tensor, target_shape.clone())
    };
    let mask = if mask.layout().shape() == &target_shape {
        mask
    } else {
        crate::ops::expand::expand(mask, target_shape.clone())
    };
    let value = if value.layout().shape() == &target_shape {
        value
    } else {
        crate::ops::expand::expand(value, target_shape)
    };

    (
        tensor.to_contiguous(),
        mask.to_contiguous(),
        value.to_contiguous(),
    )
}

/// Mask where for f32 (SIMD-accelerated).
pub fn mask_where_f32(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let dtype = tensor.dtype();
        let (tensor, mask, value) = broadcast_three(tensor, mask, value);
        let shape = tensor.layout().shape().clone();
        let len = tensor.storage::<f32>().len();
        let mut out = uninit_vec::<f32>(len);
        crate::simd::mask_where_f32(tensor.storage(), mask.bytes(), value.storage(), &mut out);
        FlexTensor::new(Bytes::from_elems(out), Layout::contiguous(shape), dtype)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_where::<f32>(tensor, mask, value)
    }
}

/// Mask where for f64 (SIMD-accelerated).
pub fn mask_where_f64(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let dtype = tensor.dtype();
        let (tensor, mask, value) = broadcast_three(tensor, mask, value);
        let shape = tensor.layout().shape().clone();
        let len = tensor.storage::<f64>().len();
        let mut out = uninit_vec::<f64>(len);
        crate::simd::mask_where_f64(tensor.storage(), mask.bytes(), value.storage(), &mut out);
        FlexTensor::new(Bytes::from_elems(out), Layout::contiguous(shape), dtype)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_where::<f64>(tensor, mask, value)
    }
}

/// Mask where for f16.
pub fn mask_where_f16(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    mask_where::<f16>(tensor, mask, value)
}

/// Mask where for bf16.
pub fn mask_where_bf16(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    mask_where::<bf16>(tensor, mask, value)
}

/// Mask where for i64 (SIMD-accelerated).
pub fn mask_where_i64(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let dtype = tensor.dtype();
        let (tensor, mask, value) = broadcast_three(tensor, mask, value);
        let shape = tensor.layout().shape().clone();
        let len = tensor.storage::<i64>().len();
        let mut out = uninit_vec::<i64>(len);
        crate::simd::mask_where_i64(tensor.storage(), mask.bytes(), value.storage(), &mut out);
        FlexTensor::new(Bytes::from_elems(out), Layout::contiguous(shape), dtype)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_where::<i64>(tensor, mask, value)
    }
}

/// Mask where for bool tensors (SIMD-accelerated).
pub fn mask_where_bool(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    // Preserve the input tensor's bool dtype for the output.
    let out_dtype = burn_std::BoolDType::from(tensor.dtype());
    #[cfg(feature = "simd")]
    {
        let (tensor, mask, value) = broadcast_three(tensor, mask, value);
        let shape = tensor.layout().shape().clone();
        let len = tensor.bytes().len();
        let mut out = uninit_vec::<u8>(len);
        crate::simd::mask_where_u8(tensor.bytes(), mask.bytes(), value.bytes(), &mut out);
        crate::ops::comparison::make_bool_tensor(out, shape, out_dtype)
    }
    #[cfg(not(feature = "simd"))]
    {
        let (tensor, mask, value) = broadcast_three(tensor, mask, value);
        let shape = tensor.layout().shape().clone();
        let tensor_data: &[u8] = tensor.bytes();
        let mask_data: &[u8] = mask.bytes();
        let value_data: &[u8] = value.bytes();
        let result: Vec<u8> = tensor_data
            .iter()
            .zip(mask_data.iter())
            .zip(value_data.iter())
            .map(|((&t, &m), &v)| if m != 0 { v } else { t })
            .collect();
        crate::ops::comparison::make_bool_tensor(result, shape, out_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_mask_fill_f32() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));

        let result = mask_fill_f32(tensor, mask, 0.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_mask_fill_2d() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, false, true], [2, 2]));

        let result = mask_fill_f32(tensor, mask, -1.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![-1.0, 2.0, 3.0, -1.0]);
    }

    #[test]
    fn test_mask_where_f32() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));
        let value = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], [4]));

        let result = mask_where_f32(tensor, mask, value);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn test_mask_fill_broadcast() {
        // Tensor [2, 3], mask [3] (broadcasts to [2, 3])
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true], [3]));

        let result = mask_fill_f32(tensor, mask, 0.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // Columns 0 and 2 should be filled with 0
        assert_eq!(data, vec![0.0, 2.0, 0.0, 0.0, 5.0, 0.0]);
    }

    // === Non-contiguous tensor tests ===

    #[test]
    fn test_mask_fill_transposed_tensor() {
        // [[1, 2], [3, 4]] transposed -> [[1, 3], [2, 4]]
        // Mask [[T, F], [F, T]]
        // Result: [[0, 3], [2, 0]]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let tensor = tensor.transpose(0, 1);
        assert!(!tensor.is_contiguous());

        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, false, true], [2, 2]));
        let result = mask_fill_f32(tensor, mask, 0.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0.0, 3.0, 2.0, 0.0]);
    }

    #[test]
    fn test_mask_fill_flipped_tensor() {
        // [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // Mask [T, T, F, F]
        // Result: [0, 0, 2, 1]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let tensor = crate::ops::flip::flip(tensor, &[0]);
        assert!(tensor.layout().strides()[0] < 0);

        let mask = FlexTensor::from_data(TensorData::new(vec![true, true, false, false], [4]));
        let result = mask_fill_f32(tensor, mask, 0.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0.0, 0.0, 2.0, 1.0]);
    }

    #[test]
    fn test_mask_fill_flipped_mask() {
        // Tensor [1, 2, 3, 4]
        // Mask [F, F, T, T] flipped -> [T, T, F, F]
        // Result: [0, 0, 3, 4]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![false, false, true, true], [4]));
        let mask = crate::ops::flip::flip(mask, &[0]);
        assert!(mask.layout().strides()[0] < 0);

        let result = mask_fill_f32(tensor, mask, 0.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mask_where_flipped_2d() {
        // [[1, 2], [3, 4]] with axis 0 flipped -> [[3, 4], [1, 2]]
        // Mask [[T, F], [F, T]]
        // Value [[10, 20], [30, 40]]
        // Result: [[10, 4], [1, 40]]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let tensor = crate::ops::flip::flip(tensor, &[0]);
        assert!(tensor.layout().strides()[0] < 0);

        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, false, true], [2, 2]));
        let value = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], [2, 2]));

        let result = mask_where_f32(tensor, mask, value);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![10.0, 4.0, 1.0, 40.0]);
    }

    #[test]
    fn test_mask_fill_both_flipped() {
        // Both tensor and mask flipped
        // Tensor [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // Mask [T, F, T, F] flipped -> [F, T, F, T]
        // Result: [4, 0, 2, 0]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let tensor = crate::ops::flip::flip(tensor, &[0]);
        assert!(tensor.layout().strides()[0] < 0);

        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));
        let mask = crate::ops::flip::flip(mask, &[0]);
        assert!(mask.layout().strides()[0] < 0);

        let result = mask_fill_f32(tensor, mask, 0.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![4.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_mask_fill_narrowed_tensor() {
        // Tensor [1, 2, 3, 4, 5, 6] narrowed to [2, 3, 4, 5]
        // Mask [T, F, F, T]
        // Result: [0, 3, 4, 0]
        let tensor =
            FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [6]));
        let tensor = tensor.narrow(0, 1, 4);

        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, false, true], [4]));
        let result = mask_fill_f32(tensor, mask, 0.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn test_mask_fill_i64_flipped() {
        // Integer tensor with flip
        let tensor = FlexTensor::from_data(TensorData::new(vec![10i64, 20, 30, 40], [4]));
        let tensor = crate::ops::flip::flip(tensor, &[0]);
        assert!(tensor.layout().strides()[0] < 0);

        // [40, 30, 20, 10] with mask [T, F, T, F] -> [-1, 30, -1, 10]
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true, false], [4]));
        let result = mask_fill_i64(tensor, mask, -1);
        let data: Vec<i64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![-1, 30, -1, 10]);
    }
}

//! Mask operations for conditional element replacement.

use alloc::vec::Vec;
use burn_backend::Element;
use burn_std::{Bytes, bf16, f16};

use crate::{FlexTensor, Layout};

/// Check if a layout is contiguous and not broadcasted.
#[inline]
fn is_contiguous_non_broadcast(layout: &Layout) -> bool {
    layout.is_contiguous()
        && !layout
            .strides()
            .iter()
            .zip(layout.shape().iter())
            .any(|(&stride, &dim)| dim > 1 && stride == 0)
}

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

    // Broadcast mask to tensor shape if needed (zero-copy layout metadata adjustment)
    let (mut tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);

    // In-place fast path: if input tensor storage is unique, contiguous, and non-broadcast
    if tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()) {
        let t_offset = tensor.layout().start_offset();
        let numel = tensor.layout().num_elements();
        if mask.layout().is_contiguous() {
            let m_offset = mask.layout().start_offset();
            let mask_bytes = &mask.bytes()[m_offset..m_offset + numel];
            let tensor_slice = &mut tensor.storage_mut::<T>()[t_offset..t_offset + numel];
            for (t, &m) in tensor_slice.iter_mut().zip(mask_bytes) {
                if m != 0 {
                    *t = value;
                }
            }
            return tensor;
        } else if let Some(nest) = crate::zip::collapse_for_zip(tensor.layout(), mask.layout()) {
            let mask_bytes = mask.bytes();
            let tensor_data: &mut [T] = tensor.storage_mut();
            crate::zip::zip_apply_inplace(&nest, tensor_data, mask_bytes, |t, m| {
                if m != 0 { value } else { t }
            });
            return tensor;
        } else {
            let t_layout = tensor.layout().clone();
            let m_layout = mask.layout().clone();
            let t_iter = crate::strided_index::StridedIter::new(&t_layout);
            let m_iter = crate::strided_index::StridedIter::new(&m_layout);
            let mask_bytes = mask.bytes();
            let tensor_data: &mut [T] = tensor.storage_mut();
            for (ti, mi) in t_iter.zip(m_iter) {
                if mask_bytes[mi] != 0 {
                    tensor_data[ti] = value;
                }
            }
            return tensor;
        }
    }

    // Out-of-place path without unconditional to_contiguous()
    let shape = tensor.layout().shape().clone();
    let numel = shape.num_elements();
    let mask_bytes: &[u8] = mask.bytes();
    let tensor_data: &[T] = tensor.storage();

    if let Some(result) = crate::zip::zip_map(
        tensor_data,
        tensor.layout(),
        mask_bytes,
        mask.layout(),
        |elem, m| if m != 0 { value } else { elem },
    ) {
        FlexTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), dtype)
    } else {
        let t_iter = crate::strided_index::StridedIter::new(tensor.layout());
        let m_iter = crate::strided_index::StridedIter::new(mask.layout());
        let mut result = Vec::with_capacity(numel);
        for (ti, mi) in t_iter.zip(m_iter) {
            result.push(if mask_bytes[mi] != 0 {
                value
            } else {
                tensor_data[ti]
            });
        }
        FlexTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), dtype)
    }
}

/// Mask fill for f32 (SIMD-accelerated when out-of-place contiguous, in-place when unique).
pub fn mask_fill_f32(tensor: FlexTensor, mask: FlexTensor, value: f32) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);
        if tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()) {
            return mask_fill(tensor, mask, value);
        }
        if tensor.layout().is_contiguous() && mask.layout().is_contiguous() {
            let shape = tensor.layout().shape().clone();
            let len = shape.num_elements();
            let t_off = tensor.layout().start_offset();
            let m_off = mask.layout().start_offset();
            let mut out = uninit_vec::<f32>(len);
            crate::simd::mask_fill_f32(
                &tensor.storage()[t_off..t_off + len],
                &mask.bytes()[m_off..m_off + len],
                value,
                &mut out,
            );
            return FlexTensor::new(
                Bytes::from_elems(out),
                Layout::contiguous(shape),
                tensor.dtype(),
            );
        }
        mask_fill(tensor, mask, value)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_fill(tensor, mask, value)
    }
}

/// Mask fill for f64 (SIMD-accelerated when out-of-place contiguous, in-place when unique).
pub fn mask_fill_f64(tensor: FlexTensor, mask: FlexTensor, value: f64) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);
        if tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()) {
            return mask_fill(tensor, mask, value);
        }
        if tensor.layout().is_contiguous() && mask.layout().is_contiguous() {
            let shape = tensor.layout().shape().clone();
            let len = shape.num_elements();
            let t_off = tensor.layout().start_offset();
            let m_off = mask.layout().start_offset();
            let mut out = uninit_vec::<f64>(len);
            crate::simd::mask_fill_f64(
                &tensor.storage()[t_off..t_off + len],
                &mask.bytes()[m_off..m_off + len],
                value,
                &mut out,
            );
            return FlexTensor::new(
                Bytes::from_elems(out),
                Layout::contiguous(shape),
                tensor.dtype(),
            );
        }
        mask_fill(tensor, mask, value)
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

/// Mask fill for i64 (SIMD-accelerated when out-of-place contiguous, in-place when unique).
pub fn mask_fill_i64(tensor: FlexTensor, mask: FlexTensor, value: i64) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);
        if tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()) {
            return mask_fill(tensor, mask, value);
        }
        if tensor.layout().is_contiguous() && mask.layout().is_contiguous() {
            let shape = tensor.layout().shape().clone();
            let len = shape.num_elements();
            let t_off = tensor.layout().start_offset();
            let m_off = mask.layout().start_offset();
            let mut out = uninit_vec::<i64>(len);
            crate::simd::mask_fill_i64(
                &tensor.storage()[t_off..t_off + len],
                &mask.bytes()[m_off..m_off + len],
                value,
                &mut out,
            );
            return FlexTensor::new(
                Bytes::from_elems(out),
                Layout::contiguous(shape),
                tensor.dtype(),
            );
        }
        mask_fill(tensor, mask, value)
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

/// Mask fill for bool tensors (SIMD-accelerated when out-of-place contiguous, in-place when unique).
pub fn mask_fill_bool(tensor: FlexTensor, mask: FlexTensor, value: bool) -> FlexTensor {
    let out_dtype = burn_std::BoolDType::from(tensor.dtype());
    #[cfg(feature = "simd")]
    {
        let (tensor, mask) = crate::ops::expand::broadcast_binary(tensor, mask);
        if tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()) {
            return mask_fill::<u8>(tensor, mask, value as u8);
        }
        if tensor.layout().is_contiguous() && mask.layout().is_contiguous() {
            let shape = tensor.layout().shape().clone();
            let len = shape.num_elements();
            let t_off = tensor.layout().start_offset();
            let m_off = mask.layout().start_offset();
            let mut out = uninit_vec::<u8>(len);
            crate::simd::mask_fill_u8(
                &tensor.bytes()[t_off..t_off + len],
                &mask.bytes()[m_off..m_off + len],
                value as u8,
                &mut out,
            );
            return crate::ops::comparison::make_bool_tensor(out, shape, out_dtype);
        }
        mask_fill::<u8>(tensor, mask, value as u8)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_fill::<u8>(tensor, mask, value as u8)
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
    let (mut tensor, mask, mut value) = broadcast_three(tensor, mask, value);

    let shape = tensor.layout().shape().clone();
    let numel = shape.num_elements();

    // In-place fast path: if tensor (val_false) is unique, contiguous, and non-broadcast
    if tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()) {
        let t_offset = tensor.layout().start_offset();
        if mask.layout().is_contiguous() && value.layout().is_contiguous() {
            let m_offset = mask.layout().start_offset();
            let v_offset = value.layout().start_offset();
            let mask_bytes = &mask.bytes()[m_offset..m_offset + numel];
            let value_slice = &value.storage::<T>()[v_offset..v_offset + numel];
            let tensor_slice = &mut tensor.storage_mut::<T>()[t_offset..t_offset + numel];
            for (i, &m) in mask_bytes.iter().enumerate() {
                if m != 0 {
                    tensor_slice[i] = value_slice[i];
                }
            }
            return tensor;
        } else if let Some(nest) =
            crate::zip::collapse_for_zip3(tensor.layout(), mask.layout(), value.layout())
        {
            let mask_bytes = mask.bytes();
            let value_data: &[T] = value.storage();
            let tensor_data: &mut [T] = tensor.storage_mut();
            crate::zip::zip3_apply_inplace(
                &nest,
                tensor_data,
                mask_bytes,
                value_data,
                |t, m, v| {
                    if m != 0 { v } else { t }
                },
            );
            return tensor;
        } else {
            let t_layout = tensor.layout().clone();
            let m_layout = mask.layout().clone();
            let v_layout = value.layout().clone();
            let t_iter = crate::strided_index::StridedIter::new(&t_layout);
            let m_iter = crate::strided_index::StridedIter::new(&m_layout);
            let v_iter = crate::strided_index::StridedIter::new(&v_layout);
            let mask_bytes = mask.bytes();
            let value_data: &[T] = value.storage();
            let tensor_data: &mut [T] = tensor.storage_mut();
            for ((ti, mi), vi) in t_iter.zip(m_iter).zip(v_iter) {
                if mask_bytes[mi] != 0 {
                    tensor_data[ti] = value_data[vi];
                }
            }
            return tensor;
        }
    }

    // In-place fast path: if value (val_true) is unique, contiguous, and non-broadcast
    if value.is_unique() && is_contiguous_non_broadcast(value.layout()) {
        let v_offset = value.layout().start_offset();
        if mask.layout().is_contiguous() && tensor.layout().is_contiguous() {
            let m_offset = mask.layout().start_offset();
            let t_offset = tensor.layout().start_offset();
            let mask_bytes = &mask.bytes()[m_offset..m_offset + numel];
            let tensor_slice = &tensor.storage::<T>()[t_offset..t_offset + numel];
            let value_slice = &mut value.storage_mut::<T>()[v_offset..v_offset + numel];
            for (i, &m) in mask_bytes.iter().enumerate() {
                if m == 0 {
                    value_slice[i] = tensor_slice[i];
                }
            }
            return value;
        } else if let Some(nest) =
            crate::zip::collapse_for_zip3(value.layout(), mask.layout(), tensor.layout())
        {
            let mask_bytes = mask.bytes();
            let tensor_data: &[T] = tensor.storage();
            let value_data: &mut [T] = value.storage_mut();
            crate::zip::zip3_apply_inplace(
                &nest,
                value_data,
                mask_bytes,
                tensor_data,
                |v, m, t| {
                    if m != 0 { v } else { t }
                },
            );
            return value;
        } else {
            let v_layout = value.layout().clone();
            let m_layout = mask.layout().clone();
            let t_layout = tensor.layout().clone();
            let v_iter = crate::strided_index::StridedIter::new(&v_layout);
            let m_iter = crate::strided_index::StridedIter::new(&m_layout);
            let t_iter = crate::strided_index::StridedIter::new(&t_layout);
            let mask_bytes = mask.bytes();
            let tensor_data: &[T] = tensor.storage();
            let value_data: &mut [T] = value.storage_mut();
            for ((vi, mi), ti) in v_iter.zip(m_iter).zip(t_iter) {
                if mask_bytes[mi] == 0 {
                    value_data[vi] = tensor_data[ti];
                }
            }
            return value;
        }
    }

    // Out-of-place path using 3-way zip
    let mask_bytes: &[u8] = mask.bytes();
    let tensor_data: &[T] = tensor.storage();
    let value_data: &[T] = value.storage();

    if let Some(result) = crate::zip::zip3_map(
        mask_bytes,
        mask.layout(),
        tensor_data,
        tensor.layout(),
        value_data,
        value.layout(),
        |m, t, v| if m != 0 { v } else { t },
    ) {
        FlexTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), dtype)
    } else {
        let m_iter = crate::strided_index::StridedIter::new(mask.layout());
        let t_iter = crate::strided_index::StridedIter::new(tensor.layout());
        let v_iter = crate::strided_index::StridedIter::new(value.layout());
        let mut result = Vec::with_capacity(numel);
        for ((mi, ti), vi) in m_iter.zip(t_iter).zip(v_iter) {
            result.push(if mask_bytes[mi] != 0 {
                value_data[vi]
            } else {
                tensor_data[ti]
            });
        }
        FlexTensor::new(Bytes::from_elems(result), Layout::contiguous(shape), dtype)
    }
}

/// Helper to broadcast three tensors to the same shape without copying.
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

    (tensor, mask, value)
}

/// Mask where for f32 (SIMD-accelerated when out-of-place contiguous, in-place when unique).
pub fn mask_where_f32(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let (tensor, mask, value) = broadcast_three(tensor, mask, value);
        if (tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()))
            || (value.is_unique() && is_contiguous_non_broadcast(value.layout()))
        {
            return mask_where::<f32>(tensor, mask, value);
        }
        if tensor.layout().is_contiguous()
            && mask.layout().is_contiguous()
            && value.layout().is_contiguous()
        {
            let shape = tensor.layout().shape().clone();
            let len = shape.num_elements();
            let t_off = tensor.layout().start_offset();
            let m_off = mask.layout().start_offset();
            let v_off = value.layout().start_offset();
            let mut out = uninit_vec::<f32>(len);
            crate::simd::mask_where_f32(
                &tensor.storage()[t_off..t_off + len],
                &mask.bytes()[m_off..m_off + len],
                &value.storage()[v_off..v_off + len],
                &mut out,
            );
            return FlexTensor::new(
                Bytes::from_elems(out),
                Layout::contiguous(shape),
                tensor.dtype(),
            );
        }
        mask_where::<f32>(tensor, mask, value)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_where::<f32>(tensor, mask, value)
    }
}

/// Mask where for f64 (SIMD-accelerated when out-of-place contiguous, in-place when unique).
pub fn mask_where_f64(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let (tensor, mask, value) = broadcast_three(tensor, mask, value);
        if (tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()))
            || (value.is_unique() && is_contiguous_non_broadcast(value.layout()))
        {
            return mask_where::<f64>(tensor, mask, value);
        }
        if tensor.layout().is_contiguous()
            && mask.layout().is_contiguous()
            && value.layout().is_contiguous()
        {
            let shape = tensor.layout().shape().clone();
            let len = shape.num_elements();
            let t_off = tensor.layout().start_offset();
            let m_off = mask.layout().start_offset();
            let v_off = value.layout().start_offset();
            let mut out = uninit_vec::<f64>(len);
            crate::simd::mask_where_f64(
                &tensor.storage()[t_off..t_off + len],
                &mask.bytes()[m_off..m_off + len],
                &value.storage()[v_off..v_off + len],
                &mut out,
            );
            return FlexTensor::new(
                Bytes::from_elems(out),
                Layout::contiguous(shape),
                tensor.dtype(),
            );
        }
        mask_where::<f64>(tensor, mask, value)
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

/// Mask where for i64 (SIMD-accelerated when out-of-place contiguous, in-place when unique).
pub fn mask_where_i64(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    #[cfg(feature = "simd")]
    {
        let (tensor, mask, value) = broadcast_three(tensor, mask, value);
        if (tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()))
            || (value.is_unique() && is_contiguous_non_broadcast(value.layout()))
        {
            return mask_where::<i64>(tensor, mask, value);
        }
        if tensor.layout().is_contiguous()
            && mask.layout().is_contiguous()
            && value.layout().is_contiguous()
        {
            let shape = tensor.layout().shape().clone();
            let len = shape.num_elements();
            let t_off = tensor.layout().start_offset();
            let m_off = mask.layout().start_offset();
            let v_off = value.layout().start_offset();
            let mut out = uninit_vec::<i64>(len);
            crate::simd::mask_where_i64(
                &tensor.storage()[t_off..t_off + len],
                &mask.bytes()[m_off..m_off + len],
                &value.storage()[v_off..v_off + len],
                &mut out,
            );
            return FlexTensor::new(
                Bytes::from_elems(out),
                Layout::contiguous(shape),
                tensor.dtype(),
            );
        }
        mask_where::<i64>(tensor, mask, value)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_where::<i64>(tensor, mask, value)
    }
}

/// Mask where for bool tensors (SIMD-accelerated when out-of-place contiguous, in-place when unique).
pub fn mask_where_bool(tensor: FlexTensor, mask: FlexTensor, value: FlexTensor) -> FlexTensor {
    let out_dtype = burn_std::BoolDType::from(tensor.dtype());
    #[cfg(feature = "simd")]
    {
        let (tensor, mask, value) = broadcast_three(tensor, mask, value);
        if (tensor.is_unique() && is_contiguous_non_broadcast(tensor.layout()))
            || (value.is_unique() && is_contiguous_non_broadcast(value.layout()))
        {
            return mask_where::<u8>(tensor, mask, value);
        }
        if tensor.layout().is_contiguous()
            && mask.layout().is_contiguous()
            && value.layout().is_contiguous()
        {
            let shape = tensor.layout().shape().clone();
            let len = shape.num_elements();
            let t_off = tensor.layout().start_offset();
            let m_off = mask.layout().start_offset();
            let v_off = value.layout().start_offset();
            let mut out = uninit_vec::<u8>(len);
            crate::simd::mask_where_u8(
                &tensor.bytes()[t_off..t_off + len],
                &mask.bytes()[m_off..m_off + len],
                &value.bytes()[v_off..v_off + len],
                &mut out,
            );
            return crate::ops::comparison::make_bool_tensor(out, shape, out_dtype);
        }
        mask_where::<u8>(tensor, mask, value)
    }
    #[cfg(not(feature = "simd"))]
    {
        mask_where::<u8>(tensor, mask, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_mask_fill_inplace_pointer_reuse() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, vec![4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![false, true, false, true], vec![4]));

        let initial_ptr = tensor.bytes().as_ptr();
        let result = mask_fill_f32(tensor, mask, 99.0);

        // Asserts in-place mutation without reallocating buffer
        assert!(core::ptr::eq(initial_ptr, result.bytes().as_ptr()));
        let values: Vec<f32> = result.into_data().try_into_vec().unwrap();
        assert_eq!(values, vec![1.0, 99.0, 3.0, 99.0]);
    }

    #[test]
    fn test_mask_fill_inplace_with_start_offset() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, vec![4, 4]));
        // Sliced view with start_offset = 4
        let sliced = tensor.narrow(0, 1, 2);
        drop(tensor); // Release parent tensor so sliced is unique
        assert!(sliced.is_unique());
        assert_eq!(sliced.layout().start_offset(), 4);
        assert!(sliced.is_contiguous());

        let mask = FlexTensor::from_data(TensorData::new(
            vec![true, false, false, true, false, true, false, false],
            vec![2, 4],
        ));

        let initial_ptr = sliced.bytes().as_ptr();
        let result = mask_fill_f32(sliced, mask, -1.0);

        // Must mutate in-place preserving start_offset
        assert!(core::ptr::eq(initial_ptr, result.bytes().as_ptr()));
        assert_eq!(result.layout().start_offset(), 4);
        let values: Vec<f32> = result.into_data().try_into_vec().unwrap();
        assert_eq!(values, vec![-1.0, 5.0, 6.0, -1.0, 8.0, -1.0, 10.0, 11.0]);
    }

    #[test]
    fn test_mask_where_inplace_pointer_reuse() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]));
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true, false], vec![4]));
        let value =
            FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]));

        let initial_ptr = tensor.bytes().as_ptr();
        let result = mask_where_f32(tensor, mask, value);

        assert!(core::ptr::eq(initial_ptr, result.bytes().as_ptr()));
        let values: Vec<f32> = result.into_data().try_into_vec().unwrap();
        assert_eq!(values, vec![10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn test_mask_where_mixed_broadcasting() {
        // [2, 1] tensor
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0], vec![2, 1]));
        // [1, 3] mask
        let mask = FlexTensor::from_data(TensorData::new(vec![true, false, true], vec![1, 3]));
        // [2, 3] value
        let value = FlexTensor::from_data(TensorData::new(
            vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![2, 3],
        ));

        let result = mask_where_f32(tensor, mask, value);
        assert_eq!(result.layout().shape().to_vec(), vec![2, 3]);

        let values: Vec<f32> = result.into_data().try_into_vec().unwrap();
        assert_eq!(values, vec![10.0, 1.0, 30.0, 40.0, 2.0, 60.0]);
    }
}

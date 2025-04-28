use core::any::TypeId;

use crate::{Bytes, Element};
use alloc::vec::Vec;

use super::{
    QParams, QuantInputType, QuantLevel, QuantMode, QuantScheme, QuantizationStrategy,
    SymmetricQuantization, pack_i8s_to_u32s, unpack_u32s_to_i8s,
};

/// Quantized data bytes representation.
///
/// # Notes
/// 1) The quantized values are packed into 32-bit unsigned integers. For example, int8
///    quantized values pack 4 grouped values into a single `u32`. When unpacking these values,
///    we make sure to retrieve only the meaningful values (and ignore the alignment padding).
/// 2) Quantization parameters are appended to the tensor data.
///    As such, the last bytes always correspond to the scale parameter.
///    If the quantization scheme includes an offset (zero-point) parameter, it is next to last.
pub struct QuantizedBytes {
    /// The quantized values and quantization parameters represented as bytes.
    pub bytes: Bytes,
    /// The quantization scheme.
    pub scheme: QuantScheme,
    /// The number of quantized elements.
    pub num_elements: usize,
}

impl QuantizedBytes {
    /// Creates a new quantized bytes representation.
    pub fn new<E: Element>(value: Vec<E>, strategy: QuantizationStrategy) -> Self {
        let mut bytes: Bytes;
        let num_elements = value.len();
        let scheme = strategy.scheme();

        match strategy {
            QuantizationStrategy::PerTensorSymmetricInt8(quant) => {
                if TypeId::of::<E>() == TypeId::of::<i8>() {
                    // Re-interpret `Vec<E>` as `Vec<i8>` with `Vec::from_raw_parts`
                    let u32s = pack_i8s_to_u32s(bytemuck::allocation::cast_vec(value));
                    bytes = Bytes::from_elems(u32s);
                } else {
                    panic!("Invalid quantized type");
                }
                let scale_bytes = bytemuck::bytes_of(&quant.scale);
                bytes.extend_from_byte_slice_aligned(scale_bytes, align_of::<f32>());
            }
        }

        Self {
            bytes,
            scheme,
            num_elements,
        }
    }

    /// Returns the int8 quantized values with the quantization parameters.
    pub fn into_vec_i8(self) -> (Vec<i8>, QParams<Vec<f32>, Vec<i8>>) {
        let numel = self.num_elements;
        let (values, (qparams, num_params)) = self.split_values_off();

        let values = unpack_u32s_to_i8s(values, numel);

        // Quantization parameters are added at the end of the tensor data.
        // As such, the last bytes always correspond to the scale parameter(s).
        // If the quantization scheme includes an offset (zero-point) parameter, the value(s)
        // precede(s) the scale parameter(s) bytes.
        // For example, per-block quantization can have multiple parameters for a single tensor:
        // [offset, offset, offset, ..., scale, scale, scale, ...]
        let scale_size = core::mem::size_of::<f32>(); // scale is stored as f32
        let qparams_bytes: &[u8] = bytemuck::cast_slice(&qparams);
        let total_bytes = qparams_bytes.len();

        let scales_size = scale_size * num_params;

        let scale = bytemuck::cast_slice(&qparams_bytes[total_bytes - scales_size..]).to_vec();
        let offset = None;

        (values, QParams { scale, offset })
    }

    /// Splits the quantized values of the tensor from the quantization parameters.
    ///
    /// Returns the packed values and a newly allocated vector containing the quantization parameters.
    fn split_values_off(self) -> (Vec<u32>, (Vec<u32>, usize)) {
        // The bytes can be created either from packed u32 or existing bytes with the same representation.
        let mut values = match self.bytes.align() {
            1 => {
                let bytes = self.bytes.try_into_vec::<u8>().unwrap();
                #[cfg(target_endian = "little")]
                {
                    // SAFETY: quantized bytes representation is created from packed u32 values in little endian
                    unsafe { reinterpret_vec(bytes) }
                }
                #[cfg(target_endian = "big")]
                {
                    pack_i8s_to_u32s(bytemuck::allocation::cast_vec(bytes))
                }
            }
            4 => self.bytes.try_into_vec::<u32>().unwrap(),
            _ => unreachable!(),
        };

        let num_params = match self.scheme.level {
            QuantLevel::Tensor => 1,
        };

        let scale_size = num_params; // f32 scale is the same number of bytes as u32
        let values_end = values.len() - scale_size;

        let qparams = values.split_off(values_end);

        (values, (qparams, num_params))
    }

    /// Dequantizes the data according to its quantization scheme.
    pub fn dequantize(self) -> (Vec<f32>, QParams<Vec<f32>, Vec<i8>>) {
        match self.scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                q_type: QuantInputType::QInt8,
                ..
            } => {
                let (values, qparams) = self.into_vec_i8();
                let strategy = QuantizationStrategy::PerTensorSymmetricInt8(
                    SymmetricQuantization::init(qparams.scale[0]),
                );
                (strategy.dequantize(&values), qparams)
            }
        }
    }
}

/// Reinterprets a `Vec<T>` as a `Vec<U>` without reallocation.
///
/// # Safety
/// - The alignment of `U` must be compatible with `T`.
/// - The size of `T` must be a multiple of the size of `U`.
/// - The input `Vec<T>` must have a length that aligns with the size of `U`.
unsafe fn reinterpret_vec<T, U>(mut input: Vec<T>) -> Vec<U> {
    // Ensure alignment and size compatibility
    assert!(
        input.as_mut_ptr().align_offset(align_of::<U>()) == 0,
        "Alignment mismatch"
    );
    assert!(
        size_of::<T>() != 0 && size_of::<U>() != 0,
        "Zero-sized types not allowed"
    );
    assert!(
        input.len() * size_of::<T>() % size_of::<U>() == 0,
        "Size mismatch"
    );

    let len = input.len() * size_of::<T>() / size_of::<U>();
    let cap = input.capacity() * size_of::<T>() / size_of::<U>();
    let ptr = input.as_mut_ptr() as *mut U;

    core::mem::forget(input);

    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

#[cfg(test)]
mod tests {

    use super::*;
    use alloc::vec;

    #[test]
    fn should_pack_unpack_quantization_parameters_per_tensor_symmetric() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let scale = 0.03937008;
        let values = vec![0i8, 25, 51, 76, 102, 127];

        let q_bytes = QuantizedBytes::new(
            values.clone(),
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(scale)),
        );

        let (q_values, qparams) = q_bytes.into_vec_i8();

        assert_eq!(qparams.scale, vec![scale]);
        assert_eq!(qparams.offset, None);

        assert_eq!(q_values, values);
    }
}

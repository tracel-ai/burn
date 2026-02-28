//! Quantization data representation.

// Re-exported types
pub use cubecl_common::quant::scheme::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};

/// Alignment (in bytes) for quantization parameters in serialized tensor data.
///
/// NOTE: This is currently f32-based since scales were originally always f32.
/// With `QuantParam` now supporting different precisions (F16, BF16, etc.),
/// this alignment may need to be revisited in the future.
pub const QPARAM_ALIGN: usize = core::mem::align_of::<f32>();

use alloc::vec::Vec;
use core::any::TypeId;
use num_traits::PrimInt;
use serde::{Deserialize, Serialize};

use crate::{DType, Metadata, Shape, bytes::Bytes};

#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
/// The precision of accumulating elements.
pub enum QuantAcc {
    /// Full precision.
    #[default]
    F32,
    /// Half precision.
    F16,
    /// bfloat16 precision.
    BF16,
}

/// Specify if the output of an operation is quantized using the scheme of the input
/// or returned unquantized.
#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub enum QuantPropagation {
    /// The output is quantized using the scheme of the input.
    Propagate,
    /// The output is not quantized.
    #[default]
    Inhibit,
}

/// The quantization tensor data parameters.
#[derive(Clone, Debug)]
pub struct QParams<S> {
    /// The scaling factor.
    pub scales: S,
    /// Optional zero-points for asymmetric quantization.
    /// Used in dequantization: `(q - zero_point) * scale`
    /// Present when using asymmetric quantization schemes.
    pub zero_points: Option<S>,
}

/// A quantization parameter tensor descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QParamTensor {
    /// Start of the tensor in the buffer
    pub offset_start: usize,
    /// Offset of tensor end from the end of the buffer
    pub offset_end: usize,
    /// Metadata of the tensor
    pub metadata: Metadata,
    /// Data type of the tensor
    pub dtype: DType,
}

/// Calculate the shape of the quantization parameters for a given tensor and level
pub fn params_shape(data_shape: &Shape, level: QuantLevel) -> Shape {
    match level {
        QuantLevel::Tensor => Shape::new([1]),
        QuantLevel::Block(block_size) => {
            let mut params_shape = data_shape.clone();
            let block_size = block_size.to_dim_vec(data_shape.num_dims());

            for (shape, block_size) in params_shape.iter_mut().zip(block_size) {
                *shape = (*shape).div_ceil(block_size as usize);
            }

            params_shape
        }
    }
}

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
    pub fn new<E: bytemuck::CheckedBitPattern + bytemuck::NoUninit>(
        value: Vec<E>,
        scheme: QuantScheme,
        scales: &[f32],
    ) -> Self {
        let num_elements = value.len();
        // Only used for 8-bit quantization data comparison in tests
        if TypeId::of::<E>() != TypeId::of::<i8>() {
            panic!("Invalid quantized type");
        }

        // Re-interpret `Vec<E>` as `Vec<i8>` with `Vec::from_raw_parts`
        let i8s: Vec<i8> = bytemuck::allocation::cast_vec(value);
        let mut bytes = Bytes::from_elems(i8s);

        match scheme.level {
            QuantLevel::Tensor => {
                let scale_bytes = bytemuck::bytes_of(&scales[0]);
                bytes.extend_from_byte_slice_aligned(scale_bytes, QPARAM_ALIGN);
            }
            QuantLevel::Block(_block_size) => {
                let mut scale_bytes = Vec::with_capacity(size_of_val(scales));
                for scale in scales {
                    scale_bytes.extend_from_slice(bytemuck::bytes_of(scale));
                }
                bytes.extend_from_byte_slice_aligned(scale_bytes.as_slice(), QPARAM_ALIGN);
            }
        }

        Self {
            bytes,
            scheme,
            num_elements,
        }
    }

    /// Returns the int8 quantized values with the quantization parameters.
    pub fn into_vec_i8(self) -> (Vec<i8>, QParams<Vec<f32>>) {
        let (values, (qparams, num_params)) = self.split_values_off();

        // Quantization parameters are added at the end of the tensor data.
        // As such, the last bytes always correspond to the scale parameter(s).
        // For example, per-block quantization can have multiple parameters for a single tensor:
        // [scale, scale, scale, ...]
        let scale_size = core::mem::size_of::<f32>(); // scale is stored as f32
        let qparams_bytes: &[u8] = bytemuck::cast_slice(&qparams);
        let total_bytes = qparams_bytes.len();

        let scales_size = scale_size * num_params;

        let scales = bytemuck::cast_slice(&qparams_bytes[total_bytes - scales_size..]).to_vec();

        (values, QParams { scales, zero_points: None })
    }

    fn split_i8_values(self, num_params: usize) -> (Vec<i8>, Vec<u32>) {
        let mut values = read_bytes_to_i8(self.bytes);

        let scale_size = num_params * size_of::<f32>();
        let values_end = values.len() - scale_size;

        let qparams = values.split_off(values_end);

        let qparams = if (qparams.as_ptr() as usize).is_multiple_of(4) {
            let mut qparams = core::mem::ManuallyDrop::new(qparams);
            unsafe {
                Vec::<u32>::from_raw_parts(
                    qparams.as_mut_ptr() as _,
                    qparams.len() / 4,
                    qparams.capacity() / 4,
                )
            }
        } else {
            #[cfg(target_endian = "little")]
            {
                // SAFETY: quantized bytes representation is created from packed u32 values in little endian
                bytemuck::cast_vec(qparams)
            }
            #[cfg(target_endian = "big")]
            {
                crate::quantization::pack_i8s_to_u32s(bytemuck::cast_vec(qparams))
            }
        };
        (values, qparams)
    }

    /// Splits the quantized values of the tensor from the quantization parameters.
    ///
    /// Returns the values in i8 and a newly allocated vector containing the quantization parameters.
    fn split_values_off(self) -> (Vec<i8>, (Vec<u32>, usize)) {
        let num_params = match self.scheme.level {
            QuantLevel::Tensor => 1,
            QuantLevel::Block(block_size) => self.num_elements / block_size.num_elements(),
        };

        if let QuantStore::PackedU32(packed_dim) = self.scheme.store {
            assert_eq!(
                packed_dim, 0,
                "Packing must be on innermost dimension for splitting off values"
            );
        }

        let (values, qparams) = match self.scheme.store {
            QuantStore::Native => self.split_i8_values(num_params),
            QuantStore::PackedU32(_) => match self.scheme.value {
                QuantValue::Q8F | QuantValue::Q8S => self.split_i8_values(num_params),
                QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                    let mut values = self.bytes.try_into_vec::<u32>().unwrap();
                    let scale_size = num_params; // size of f32 same as u32
                    let values_end = values.len() - scale_size;

                    let qparams = values.split_off(values_end);
                    // Sub-byte values are unpacked as i8s for value equality tests
                    let values = unpack_q_to_i8s(&values, self.num_elements, &self.scheme.value);
                    (values, qparams)
                }
                QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => {
                    unimplemented!("Not yet supported")
                }
            },
            QuantStore::PackedNative(_) => unimplemented!("Not yet supported"),
        };

        (values, (qparams, num_params))
    }
}

fn read_bytes_to_i8(bytes: Bytes) -> Vec<i8> {
    match bytes.try_into_vec::<i8>() {
        Ok(val) => val,
        // Safety,
        //
        // `Vec<u8>` can be Re-interpreted as `Vec<i8>` since they share the same alignment.
        Err(bytes) => unsafe { core::mem::transmute::<Vec<u8>, Vec<i8>>(bytes.to_vec()) },
    }
}

/// Pack signed 8-bit integer values into a sequence of unsigned 32-bit integers.
pub fn pack_i8s_to_u32s(values: Vec<i8>) -> Vec<u32> {
    // Shift and combine groups of four 8-bit values into a u32.
    // Same as doing this:
    //     let result = (d_u8 & 0xFF) << 24 | (c_u8 & 0xFF) << 16 | (b_u8 & 0xFF) << 8 | (a_u8 & 0xFF);
    #[cfg(target_endian = "big")]
    {
        values
            .chunks(4)
            .map(|x| {
                x.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, x)| acc | (*x as u32 & 0xFF) << (i * 8))
            })
            .collect()
    }

    // The order of bytes in little endian matches the above description, we just need to
    // handle padding when the number of values is not a factor of 4
    #[cfg(target_endian = "little")]
    {
        let mut values = values;
        let remainder = values.len() % 4;
        if remainder != 0 {
            // Pad with zeros
            values.extend(core::iter::repeat_n(0, 4 - remainder));
        }

        let len = values.len() / 4;
        let capacity = values.capacity() / 4;

        // Pre-forget the old vec and re-interpret as u32
        let mut values = core::mem::ManuallyDrop::new(values);
        let ptr = values.as_mut_ptr() as *mut u32;

        unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    }
}

/// Unpack integer values into a sequence of signed 8-bit integers.
pub(crate) fn unpack_q_to_i8s<Q: PrimInt>(
    values: &[Q],
    numel: usize,
    value: &QuantValue,
) -> Vec<i8> {
    let size_store = size_of::<Q>() * 8;
    let size_quant = value.size_bits();
    let num_quants = size_store / size_quant;
    let mask = Q::from((1 << size_quant) - 1).unwrap();
    let sign_shift = 8 - size_quant; // sign extension for sub-byte values
    values
        .iter()
        .enumerate()
        .flat_map(|(i, &packed)| {
            // A single u32 could contain less than four 8-bit values...
            let n = core::cmp::min(num_quants, numel - i * num_quants);
            // Extract each 8-bit segment from u32 and cast back to i8
            // Same as doing this (when 4 values are fully packed):
            //     let a = (packed & 0xFF) as i8;
            //     let b = ((packed >> 8) & 0xFF) as i8;
            //     let c = ((packed >> 16) & 0xFF) as i8;
            //     let d = ((packed >> 24) & 0xFF) as i8;
            (0..n).map(move |i| {
                let raw = (packed >> (i * size_quant) & mask).to_u8().unwrap();
                ((raw << sign_shift) as i8) >> sign_shift
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use super::*;
    use alloc::vec;

    #[test]
    fn should_pack_i8s_to_u32() {
        let packed = pack_i8s_to_u32s(vec![-128, 2, -3, 127]);

        assert_eq!(packed, vec![2147287680]);
    }

    #[test]
    fn should_pack_i8s_to_u32_padded() {
        let packed = pack_i8s_to_u32s(vec![-128, 2, -3, 127, 55]);
        let packed_padded = pack_i8s_to_u32s(vec![-128, 2, -3, 127, 55, 0, 0, 0]);

        assert_eq!(packed, vec![2147287680, 55]);
        assert_eq!(packed, packed_padded);
    }

    #[test]
    fn should_unpack_u32s_to_i8s() {
        let unpacked = unpack_q_to_i8s(&[2147287680u32], 4, &QuantValue::Q8S);

        assert_eq!(unpacked, vec![-128, 2, -3, 127]);
    }

    #[test]
    fn should_unpack_u32s_to_i8s_padded() {
        let unpacked = unpack_q_to_i8s(&[55u32], 1, &QuantValue::Q8S);

        assert_eq!(unpacked, vec![55]);
    }

    #[test]
    fn should_unpack_u32s_to_i8s_arange() {
        let unpacked = unpack_q_to_i8s(
            &[
                0u32, 286331136, 286331153, 572657937, 572662306, 857874978, 858993459, 858993459,
                1145324612, 1145324612, 1431655748, 1431655765, 1717982549, 1717986918, 2003199590,
                2004318071,
            ],
            128,
            &QuantValue::Q4S,
        );

        assert_eq!(
            unpacked,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
            ]
        );
    }

    #[test]
    fn should_pack_unpack_quantization_parameters_per_tensor_symmetric() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let scale = 0.03937008;
        let values = vec![0i8, 25, 51, 76, 102, 127];

        let q_bytes = QuantizedBytes::new(
            values.clone(),
            QuantScheme::default()
                .with_value(QuantValue::Q8S)
                .with_store(QuantStore::Native),
            &[scale],
        );

        let (q_values, qparams) = q_bytes.into_vec_i8();

        assert_eq!(qparams.scales, vec![scale]);

        assert_eq!(q_values, values);
    }
}

use core::any::TypeId;

use crate::{Bytes, Element, quantization::unpack_q_to_i8s};
use alloc::vec::Vec;

use super::{QParams, QuantLevel, QuantScheme, QuantStore, QuantValue};

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
    pub fn new<E: Element>(value: Vec<E>, scheme: QuantScheme, scales: &[f32]) -> Self {
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
                bytes.extend_from_byte_slice_aligned(scale_bytes, align_of::<f32>());
            }
            QuantLevel::Block(_block_size) => {
                let mut scale_bytes = Vec::with_capacity(size_of_val(scales));
                for scale in scales {
                    scale_bytes.extend_from_slice(bytemuck::bytes_of(scale));
                }
                bytes.extend_from_byte_slice_aligned(scale_bytes.as_slice(), align_of::<f32>());
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

        (values, QParams { scales })
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

        let (values, qparams) = match self.scheme.store {
            QuantStore::Native => self.split_i8_values(num_params),
            QuantStore::U32 => match self.scheme.value {
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
        Err(bytes) => unsafe { core::mem::transmute(bytes.to_vec()) },
    }
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

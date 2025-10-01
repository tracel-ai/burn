use crate::quantization::QuantValue;
use alloc::vec::Vec;
use num_traits::PrimInt;

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
}

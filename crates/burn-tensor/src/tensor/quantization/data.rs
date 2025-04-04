use alloc::vec::Vec;

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

/// Unpack 32-bit unsigned integer values into a sequence of signed 8-bit integers.
pub fn unpack_u32s_to_i8s(values: Vec<u32>, numel: usize) -> Vec<i8> {
    #[cfg(target_endian = "big")]
    {
        values
            .into_iter()
            .enumerate()
            .flat_map(|(i, packed)| {
                // A single u32 could contain less than four 8-bit values...
                let n = core::cmp::min(4, numel - i * 4);
                // Extract each 8-bit segment from u32 and cast back to i8
                // Same as doing this (when 4 values are fully packed):
                //     let a = (packed & 0xFF) as i8;
                //     let b = ((packed >> 8) & 0xFF) as i8;
                //     let c = ((packed >> 16) & 0xFF) as i8;
                //     let d = ((packed >> 24) & 0xFF) as i8;
                (0..n).map(move |i| (packed >> (i * 8) & 0xFF) as i8)
            })
            .collect()
    }

    // The order of bytes in little endian matches the above description, we just need to
    // handle padding when the number of elements is not a factor of 4
    #[cfg(target_endian = "little")]
    {
        let len = values.len() * 4;
        let capacity = values.capacity() * 4;

        // Pre-forget the old vec and re-interpret as u32
        let mut values = core::mem::ManuallyDrop::new(values);
        let ptr = values.as_mut_ptr() as *mut i8;

        let mut vec = unsafe { Vec::from_raw_parts(ptr, len, capacity) };

        let padding = len - numel;
        if padding > 0 {
            vec.truncate(len - padding);
        }

        vec
    }
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
        let unpacked = unpack_u32s_to_i8s(vec![2147287680u32], 4);

        assert_eq!(unpacked, vec![-128, 2, -3, 127]);
    }

    #[test]
    fn should_unpack_u32s_to_i8s_padded() {
        let unpacked = unpack_u32s_to_i8s(vec![55u32], 1);

        assert_eq!(unpacked, vec![55]);
    }
}

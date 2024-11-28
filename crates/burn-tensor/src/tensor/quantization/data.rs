use alloc::vec::Vec;

/// Pack signed 8-bit integer values into a sequence of unsigned 32-bit integers.
///
/// # Note
/// This assumes that the bytes represent `i8` values.
pub fn pack_i8s_to_u32s(bytes: &[u8]) -> Vec<u32> {
    // Shift and combine groups of four 8-bit values into a u32.
    // Same as doing this:
    //     let result = (a_u8 & 0xFF) << 24 | (b_u8 & 0xFF) << 16 | (c_u8 & 0xFF) << 8 | (d_u8 & 0xFF);
    bytes
        .chunks(4)
        .map(|x| {
            x.iter().enumerate().fold(0u32, |acc, (i, x)| {
                acc | (*x as i8 as u32 & 0xFF) << ((3 - i) * 8)
            })
        })
        .collect()
}

/// Unpack 32-bit unsigned integer values into a sequence of signed 8-bit integers.
///
/// # Note
/// This assumes that the bytes represent `u32` values.
pub fn unpack_u32s_to_i8s(bytes: &[u8], numel: usize) -> Vec<i8> {
    bytemuck::cast_slice::<_, u32>(bytes)
        .iter()
        .enumerate()
        .flat_map(|(i, packed)| {
            // A single u32 could contain less than four 8-bit values...
            let n = core::cmp::min(4, numel - i * 4);
            // Extract each 8-bit segment from u32 and cast back to i8
            // Same as doing this (when 4 values are fully packed):
            //     let a = ((packed >> 24) & 0xFF) as i8;
            //     let b = ((packed >> 16) & 0xFF) as i8;
            //     let c = ((packed >> 8) & 0xFF) as i8;
            //     let d = (packed & 0xFF) as i8;
            (0..n).map(move |i| (packed >> ((3 - i) * 8) & 0xFF) as i8)
        })
        .collect()
}

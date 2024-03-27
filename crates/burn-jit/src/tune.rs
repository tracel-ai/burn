//! Module with tune utilities.

use std::cmp::min;

/// Anchor a number to a power of 2.
///
/// Useful when creating autotune keys.
pub fn anchor(x: usize, max: Option<usize>) -> usize {
    let exp = f32::ceil(f32::log2(x as f32)) as u32;
    let power_of_2 = 2_u32.pow(exp) as usize;
    if let Some(max) = max {
        min(power_of_2, max)
    } else {
        power_of_2
    }
}

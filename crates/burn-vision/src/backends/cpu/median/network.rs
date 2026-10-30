use macerator::{Simd, VOrd, Vector, vload_unaligned, vstore_unaligned};

pub(super) fn filter_3x3(
    input: &[f32],
    output: &mut [f32],
    height: usize,
    width: usize,
    keys: &mut Vec<i32>,
) {
    debug_assert!(height > 0 && width > 0);
    debug_assert_eq!(input.len(), height * width);
    debug_assert_eq!(input.len(), output.len());

    keys.clear();
    keys.extend(
        input
            .iter()
            .map(|value| order_key32(value.to_bits() as i32)),
    );

    // Integer min/max preserves total_cmp ordering, including signed zeros and NaN payloads.
    filter_ord(keys, bytemuck::cast_slice_mut(output), height, width);
    for value in output {
        *value = f32::from_bits(order_key32(value.to_bits() as i32) as u32);
    }
}

pub(super) fn filter_f64(
    input: &[f64],
    output: &mut [f64],
    height: usize,
    width: usize,
    keys: &mut Vec<i64>,
) {
    debug_assert!(height > 0 && width > 0);
    debug_assert_eq!(input.len(), height * width);
    debug_assert_eq!(input.len(), output.len());

    keys.clear();
    keys.extend(
        input
            .iter()
            .map(|value| order_key64(value.to_bits() as i64)),
    );
    filter_ord(keys, bytemuck::cast_slice_mut(output), height, width);
    for value in output {
        *value = f64::from_bits(order_key64(value.to_bits() as i64) as u64);
    }
}

// The sign-preserving transforms from float total_cmp are their own inverses.
#[inline(always)]
fn order_key32(bits: i32) -> i32 {
    bits ^ (((bits >> 31) as u32) >> 1) as i32
}

#[inline(always)]
fn order_key64(bits: i64) -> i64 {
    bits ^ (((bits >> 63) as u64) >> 1) as i64
}

#[macerator::with_simd]
pub(super) fn filter_ord<S: Simd, T: VOrd + Ord>(
    input: &[T],
    output: &mut [T],
    height: usize,
    width: usize,
) {
    debug_assert!(height > 0 && width > 0);
    debug_assert_eq!(input.len(), height * width);
    debug_assert_eq!(input.len(), output.len());

    let lanes = T::lanes::<S>();

    for y in 0..height {
        let rows = [
            &input[y.saturating_sub(1) * width..][..width],
            &input[y * width..][..width],
            &input[(y + 1).min(height - 1) * width..][..width],
        ];
        let output = &mut output[y * width..][..width];
        output[0] = scalar_median(rows, 0, width);

        let mut x = 1;
        while x + lanes < width {
            // SAFETY: x >= 1 and x + lanes < width keep all nine unaligned
            // loads within their rows and the store within the output row.
            unsafe {
                let values: [Vector<S, T>; 9] = core::array::from_fn(|i| {
                    vload_unaligned(rows[i / 3].as_ptr().add(x + i % 3 - 1))
                });
                let median = median9(values, T::vmin::<S>, T::vmax::<S>);
                vstore_unaligned(output.as_mut_ptr().add(x), median);
            }
            x += lanes;
        }

        for (x, value) in output.iter_mut().enumerate().skip(x) {
            *value = scalar_median(rows, x, width);
        }
    }
}

#[inline(always)]
fn scalar_median<T: Copy + Ord>(rows: [&[T]; 3], x: usize, width: usize) -> T {
    let columns = [x.saturating_sub(1), x, (x + 1).min(width - 1)];
    let values = core::array::from_fn(|i| rows[i / 3][columns[i % 3]]);
    median9(values, T::min, T::max)
}

// A 19-comparator selection network: only slot 4 is guaranteed to be sorted.
#[inline(always)]
fn median9<T: Copy>(mut values: [T; 9], min: impl Fn(T, T) -> T, max: impl Fn(T, T) -> T) -> T {
    let mut compare = |a, b| {
        let left = values[a];
        let right = values[b];
        values[a] = min(left, right);
        values[b] = max(left, right);
    };

    compare(1, 2);
    compare(4, 5);
    compare(7, 8);
    compare(0, 1);
    compare(3, 4);
    compare(6, 7);
    compare(1, 2);
    compare(4, 5);
    compare(7, 8);
    compare(0, 3);
    compare(5, 8);
    compare(4, 7);
    compare(3, 6);
    compare(1, 4);
    compare(2, 5);
    compare(4, 7);
    compare(4, 2);
    compare(6, 4);
    compare(4, 2);

    values[4]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_network_finds_all_binary_medians() {
        for bits in 0..512u32 {
            let values = core::array::from_fn(|i| ((bits >> i) & 1) as i32);
            assert_eq!(
                median9(values, i32::min, i32::max),
                i32::from(bits.count_ones() >= 5)
            );
        }
    }
}

use core::cmp::Ordering;

use burn_core::{
    backend::Flex,
    tensor::{DType, FloatDType, Int, Tensor, TensorData},
};
use burn_std::{bf16, f16};

use super::base::coordinate;

/// Select source pixels on the CPU, then gather from the original tensor so the
/// existing gather operation retains its autodiff graph and backward scatter-add.
pub(super) fn median(images: &Tensor<4>, size: usize) -> Option<Tensor<4>> {
    let dims @ [batch, channels, height, width] = images.dims();
    if dims.contains(&0) || size == 0 {
        return None;
    }
    let window_len = size.checked_mul(size)?;
    let input = images.clone().inner().try_into_primitive::<Flex>().ok()?;
    let input = input.to_contiguous();
    let dtype = input.dtype();
    let indices = match dtype {
        DType::F32 => select_indices(
            input.as_slice::<f32>()?,
            [height, width],
            size,
            window_len,
            core::convert::identity,
            f32::total_cmp,
        ),
        DType::F64 => select_indices(
            input.as_slice::<f64>()?,
            [height, width],
            size,
            window_len,
            core::convert::identity,
            f64::total_cmp,
        ),
        DType::F16 => select_indices(
            input.as_slice::<f16>()?,
            [height, width],
            size,
            window_len,
            f16::to_f32,
            f32::total_cmp,
        ),
        DType::BF16 => select_indices(
            input.as_slice::<bf16>()?,
            [height, width],
            size,
            window_len,
            bf16::to_f32,
            f32::total_cmp,
        ),
        _ => return None,
    };

    let plane_len = height * width;
    let shape = [batch, channels, plane_len];
    let device = images.device().without_autodiff();
    // Indices address one channel plane, so a narrow user-selected default int
    // dtype must not truncate them. Flex supports both widths.
    let indices: Tensor<3, Int> = if plane_len <= i32::MAX as usize {
        Tensor::from_data(
            TensorData::new(indices.into_iter().map(|i| i as i32).collect(), shape),
            (&device, DType::I32),
        )
    } else {
        Tensor::from_data(
            TensorData::new(indices.into_iter().map(|i| i as i64).collect(), shape),
            (&device, DType::I64),
        )
    };
    let output = images
        .clone()
        .reshape(shape)
        .gather(2, indices)
        .reshape(dims);
    Some(match dtype {
        // Match Flex half sorting's F32 roundtrip, including NaN conversion.
        DType::F16 | DType::BF16 => output.cast(FloatDType::F32).cast(FloatDType::from(dtype)),
        _ => output,
    })
}

fn select_indices<E: Copy, V: Copy>(
    input: &[E],
    [height, width]: [usize; 2],
    size: usize,
    window_len: usize,
    convert: impl Fn(E) -> V + Copy,
    compare: impl Fn(&V, &V) -> Ordering + Copy,
) -> Vec<usize> {
    let plane_len = height * width;
    let middle = window_len / 2;
    let mut indices = Vec::with_capacity(input.len());
    let mut scratch = Vec::with_capacity(window_len);
    for plane in input.chunks_exact(plane_len) {
        let window = Window {
            plane,
            height,
            width,
            size,
        };
        for y in 0..height {
            for x in 0..width {
                window.copy(y, x, convert, &mut scratch);
                let &mut (mut source, value) = scratch
                    .select_nth_unstable_by(middle, |a, b| compare(&a.1, &b.1))
                    .1;
                if scratch
                    .iter()
                    .any(|(other, v)| *other != source && compare(v, &value) == Ordering::Equal)
                {
                    // Flex sort_with_indices uses an unstable value-only sort.
                    // Restore window order before reproducing its tie routing.
                    window.copy(y, x, convert, &mut scratch);
                    scratch.sort_unstable_by(|a, b| compare(&a.1, &b.1));
                    source = scratch[middle].0;
                }
                indices.push(source);
            }
        }
    }
    indices
}

struct Window<'a, E> {
    plane: &'a [E],
    height: usize,
    width: usize,
    size: usize,
}

impl<E: Copy> Window<'_, E> {
    fn copy<V>(&self, y: usize, x: usize, convert: impl Fn(E) -> V, scratch: &mut Vec<(usize, V)>) {
        scratch.clear();
        let reach = self.size / 2;
        for dy in 0..self.size {
            let row = coordinate(y, dy, reach, self.height) * self.width;
            for dx in 0..self.size {
                let source = row + coordinate(x, dx, reach, self.width);
                scratch.push((source, convert(self.plane[source])));
            }
        }
    }
}

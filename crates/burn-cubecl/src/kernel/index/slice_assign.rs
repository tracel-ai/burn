use crate::{CubeRuntime, element::CubeElement, tensor::CubeTensor};
use cubecl::std::{FastDivmod, FastDivmodArgs};
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use std::ops::Range;

#[inline]
fn clamp_range(dim: usize, r: &Range<usize>) -> (usize, usize) {
    let s = r.start.min(dim);
    let e = r.end.min(dim);
    if e < s { (s, s) } else { (s, e) }
}

#[cube(launch_unchecked)]
fn slice_assign_kernel<E: CubePrimitive>(
    input: &mut Tensor<Line<E>>,
    value: &Tensor<Line<E>>,
    slice_shape: Sequence<FastDivmod>,
    slice_offsets: Sequence<u32>,
) {
    if ABSOLUTE_POS >= value.len() {
        terminate!()
    }

    let rank = comptime!(slice_shape.len());

    let line_size = input.line_size();
    let mut offset_remainder = ABSOLUTE_POS * line_size;
    let mut offset_input = 0;
    let mut offset_value = 0;

    let mut i = comptime![0];

    #[allow(clippy::explicit_counter_loop)]
    #[unroll]
    for _ in 0..rank {
        let dim = comptime![rank - i - 1];
        let (rem, offset_local) = slice_shape.index(dim).div_mod(offset_remainder);

        let range_start = *slice_offsets.index(dim);
        let offset_local_input = offset_local + range_start;

        offset_value += offset_local * value.stride(dim);
        offset_input += offset_local_input * input.stride(dim);
        offset_remainder = rem;

        comptime![i += 1;]
    }

    input[offset_input / line_size] = value[offset_value / line_size];
}

pub(crate) fn slice_assign<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    indices: &[Range<usize>],
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    let client = tensor.client.clone();
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };
    let ndims = tensor.shape.num_dims();

    // --- Vectorization (line size) decision using clamped last-dim slice ---
    let line_size = if tensor.strides[ndims - 1] == 1 && value.strides[ndims - 1] == 1 {
        let last_dim = tensor.shape.dims[ndims - 1];
        let last_req = indices.get(ndims - 1).cloned().unwrap_or(Range {
            start: 0,
            end: last_dim,
        });
        let (start, end) = clamp_range(last_dim, &last_req);
        let shape = end.saturating_sub(start);
        let offset = start;

        *R::supported_line_sizes()
            .iter()
            .filter(|it| {
                let it = **it as usize;
                shape % it == 0
                    && strides_compatible(&tensor.strides, it)
                    && strides_compatible(&value.strides, it)
                    && offset % it == 0
            })
            .max()
            .unwrap_or(&1)
    } else {
        1
    };

    let mut shape = SequenceArg::<R, FastDivmod>::new();
    let mut offsets = SequenceArg::<R, u32>::new();

    // --- Build per-dimension shapes/offsets with clamped ranges ---
    for i in 0..ndims {
        let dimi = tensor.shape.dims[i];
        let req = indices.get(i).cloned().unwrap_or(Range {
            start: 0,
            end: dimi,
        });
        let (s, e) = clamp_range(dimi, &req);
        let length = e.saturating_sub(s);

        // Empty slice is a no-op write; skip kernel entirely.
        if length == 0 {
            return tensor;
        }

        shape.push(FastDivmodArgs::new(&client, length as u32));
        offsets.push(ScalarArg::new(s as u32));
    }

    // Nothing to write?
    if value.shape.num_elements() == 0 {
        return tensor;
    }

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(value.shape.num_elements() / line_size as usize, cube_dim);

    unsafe {
        slice_assign_kernel::launch_unchecked::<E, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(line_size),
            value.as_tensor_arg::<E>(line_size),
            shape,
            offsets,
        );
    }

    tensor
}

fn strides_compatible(strides: &[usize], vec: usize) -> bool {
    strides
        .iter()
        .all(|stride| *stride % vec == 0 || *stride == 1)
}

use crate::{CubeRuntime, element::CubeElement, tensor::CubeTensor};
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubecl_std::{FastDivmod, FastDivmodArgs};
use std::ops::Range;

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

    let line_size = if tensor.strides[ndims - 1] == 1 && value.strides[ndims - 1] == 1 {
        let last = indices.get(ndims - 1).cloned().unwrap_or(Range {
            start: 0,
            end: tensor.shape.dims[ndims - 1],
        });
        let shape = last.end - last.start;
        let offset = last.start;
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

    for i in 0..ndims {
        let range = indices.get(i).cloned().unwrap_or(Range {
            start: 0,
            end: tensor.shape.dims[i],
        });
        let start = range.start;
        let length = range.end - start;

        shape.push(FastDivmodArgs::new(&client, length as u32));
        offsets.push(ScalarArg::new(start as u32));
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

use crate::{
    CubeRuntime,
    element::CubeElement,
    kernel::utils::{linear_view, shape_divmod},
    tensor::CubeTensor,
};
use cubecl::{
    calculate_cube_count_elemwise, intrinsic,
    prelude::*,
    std::{FastDivmod, FastDivmodArgs, tensor::layout::linear::LinearView},
};

#[cube(launch_unchecked)]
fn slice_assign_kernel<E: CubePrimitive>(
    input: &mut Tensor<Line<E>>,
    value: &LinearView<Line<E>>,
    slice_shape: Sequence<FastDivmod>,
    slice_offsets: Sequence<u32>,
) {
    if !value.is_in_bounds(ABSOLUTE_POS) {
        terminate!()
    }

    let rank = comptime!(slice_shape.len());

    let line_size = input.line_size();
    let mut offset_remainder = ABSOLUTE_POS * line_size;
    let mut offset_input = 0;

    let mut i = comptime![0];

    #[allow(clippy::explicit_counter_loop)]
    #[unroll]
    for _ in 0..rank {
        let dim = comptime![rank - i - 1];
        let (rem, offset_local) = slice_shape.index(dim).div_mod(offset_remainder);

        let range_start = *slice_offsets.index(dim);
        let offset_local_input = offset_local + range_start;

        offset_input += offset_local_input * input.stride(dim);
        offset_remainder = rem;

        comptime![i += 1;]
    }

    // Value tensor is accessed linearly since it's a LinearView
    input[offset_input / line_size] = value[ABSOLUTE_POS];
}

/// Kernel for slice assign with steps
#[cube(launch_unchecked)]
fn slice_assign_with_steps_kernel<E: CubePrimitive>(
    input: &mut Tensor<E>,
    value: &LinearView<E>,
    value_shape: Sequence<FastDivmod>,
    starts: Sequence<u32>,
    ends: Sequence<u32>,
    steps: Sequence<i32>,
) {
    if !value.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = comptime![value_shape.len()];
    let mut value_offset = ABSOLUTE_POS;
    let mut input_offset = 0;

    // Calculate the input offset based on value position and slice info
    #[unroll]
    for i in 0..rank {
        // Iterate in reverse to use divmod
        let i = unwrap(i);
        let dim = comptime![rank - i - 1];
        let start = *starts.index(dim);
        let end = *ends.index(dim);
        let step = *steps.index(dim);

        let (rem, value_idx) = value_shape.index(dim).div_mod(value_offset);
        value_offset = rem;

        let input_idx = if step > 0 {
            // Forward stepping
            start + value_idx * (step as u32)
        } else if step < 0 {
            // Backward stepping - start from end-1
            // For negative steps, we iterate backwards through the selected indices
            let abs_step = (-step) as u32;
            let end_minus_1 = end - 1;
            end_minus_1 - value_idx * abs_step
        } else {
            // step == 0, shouldn't happen
            value_idx
        };

        input_offset += input_idx * input.stride(dim);
    }

    input[input_offset] = value[ABSOLUTE_POS];
}

pub(crate) fn slice_assign<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    indices: &[burn_tensor::Slice],
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    // Check if any slice has non-unit step
    let has_non_unit_step = indices.iter().any(|s| s.step != 1 && s.step != 0);

    if has_non_unit_step {
        // Use slice_assign_with_steps
        return slice_assign_with_steps::<R, E>(tensor, indices, value);
    }

    let client = tensor.client.clone();
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };
    let ndims = tensor.shape.num_dims();

    let line_size = if tensor.strides[ndims - 1] == 1 && value.strides[ndims - 1] == 1 {
        let last = indices
            .get(ndims - 1)
            .cloned()
            .unwrap_or(burn_tensor::Slice {
                start: 0,
                end: Some(tensor.shape[ndims - 1] as isize),
                step: 1,
            });
        let end = last.end.unwrap_or(tensor.shape[ndims - 1] as isize);
        let shape = (end - last.start) as usize;
        let offset = last.start as usize;
        *R::supported_line_sizes()
            .iter()
            .filter(|it| {
                let it = **it as usize;
                shape.is_multiple_of(it)
                    && strides_compatible(&tensor.strides, it)
                    && strides_compatible(&value.strides, it)
                    && offset.is_multiple_of(it)
            })
            .max()
            .unwrap_or(&1)
    } else {
        1
    };

    let mut shape = SequenceArg::<R, FastDivmod>::new();
    let mut offsets = SequenceArg::<R, u32>::new();

    for i in 0..ndims {
        let slice = indices.get(i).cloned().unwrap_or(burn_tensor::Slice {
            start: 0,
            end: Some(tensor.shape[i] as isize),
            step: 1,
        });
        let start = slice.start as usize;
        let end = slice.end.unwrap_or(tensor.shape[i] as isize);
        let length = (end - slice.start) as usize;

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
            tensor.as_tensor_arg(line_size),
            linear_view(&value, line_size),
            shape,
            offsets,
        );
    }

    tensor
}

/// Slice assign with steps support
///
/// This function handles slice assignment with arbitrary step values, including negative steps.
/// It follows NumPy/PyTorch semantics where values[i] is assigned to selected_indices[i].
///
/// For example, with s![0..6;-1] which selects indices [5,4,3,2,1,0]:
/// - values[0] goes to index 5
/// - values[1] goes to index 4
/// - etc.
pub(crate) fn slice_assign_with_steps<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    slices: &[burn_tensor::Slice],
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    // Prepare sequences for kernel
    let mut starts = SequenceArg::<R, u32>::new();
    let mut ends = SequenceArg::<R, u32>::new();
    let mut steps = SequenceArg::<R, i32>::new();

    for (dim, slice) in slices.iter().enumerate() {
        let range = slice.to_range(tensor.shape[dim]);
        starts.push(ScalarArg::new(range.start as u32));
        ends.push(ScalarArg::new(range.end as u32));
        steps.push(ScalarArg::new(slice.step as i32));
    }

    // Pad with default values if needed to match tensor dimensions
    for dim in slices.len()..tensor.shape.num_dims() {
        starts.push(ScalarArg::new(0));
        ends.push(ScalarArg::new(tensor.shape[dim] as u32));
        steps.push(ScalarArg::new(1));
    }

    // Launch kernel
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(value.shape.num_elements(), cube_dim);

    unsafe {
        slice_assign_with_steps_kernel::launch_unchecked::<E, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            linear_view(&value, 1),
            shape_divmod(&value),
            starts,
            ends,
            steps,
        );
    }

    tensor
}

fn strides_compatible(strides: &[usize], vec: usize) -> bool {
    strides
        .iter()
        .all(|stride| *stride % vec == 0 || *stride == 1)
}

/// Helper function for unwrap
#[allow(unused)]
#[cube]
fn unwrap(value: u32) -> comptime_type!(u32) {
    intrinsic!(|_| value.constant().unwrap().as_u32())
}

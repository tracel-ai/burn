use crate::{
    CubeRuntime,
    kernel::utils::{linear_view, shape_divmod},
    tensor::CubeTensor,
};
use cubecl::{
    calculate_cube_count_elemwise, intrinsic,
    prelude::*,
    std::{FastDivmod, FastDivmodArgs, tensor::layout::linear::LinearView},
};

#[cube(launch_unchecked)]
fn slice_assign_kernel<E: Numeric>(
    input: &mut Tensor<Line<E>>,
    value: &LinearView<Line<E>>,
    slice_shape: Sequence<FastDivmod<usize>>,
    slice_offsets: Sequence<usize>,
    #[define(E)] _dtype: StorageType,
) {
    if !value.is_in_bounds(ABSOLUTE_POS) {
        terminate!()
    }

    let rank = comptime!(slice_shape.len());

    let line_size = input.line_size();
    let mut offset_remainder = ABSOLUTE_POS * line_size;
    let mut offset_input = 0;

    #[allow(clippy::explicit_counter_loop)]
    #[unroll]
    for i in 0..rank {
        let dim = rank - i - 1;
        let (rem, offset_local) = slice_shape[dim].div_mod(offset_remainder);

        let range_start = slice_offsets[dim];
        let offset_local_input = offset_local + range_start;

        offset_input += offset_local_input * input.stride(dim);
        offset_remainder = rem;
    }

    // Value tensor is accessed linearly since it's a LinearView
    input[offset_input / line_size] = value[ABSOLUTE_POS];
}

/// Kernel for slice assign with steps
#[cube(launch_unchecked)]
fn slice_assign_with_steps_kernel<E: Numeric>(
    input: &mut Tensor<E>,
    value: &LinearView<E>,
    value_shape: Sequence<FastDivmod<usize>>,
    starts: Sequence<usize>,
    ends: Sequence<usize>,
    steps: Sequence<i32>,
    #[define(E)] _dtype: StorageType,
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
        let dim = rank - i - 1;
        let start = starts[dim];
        let end = ends[dim];
        let step = steps[dim];

        let (rem, value_idx) = value_shape[dim].div_mod(value_offset);
        value_offset = rem;

        let input_idx = if step > 0 {
            // Forward stepping
            start + value_idx * (step as usize)
        } else if step < 0 {
            // Backward stepping - start from end-1
            // For negative steps, we iterate backwards through the selected indices
            let abs_step = (-step) as usize;
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

pub(crate) fn slice_assign<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    indices: &[burn_backend::Slice],
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    // Check if any slice has non-unit step
    let has_non_unit_step = indices.iter().any(|s| s.step != 1 && s.step != 0);

    if has_non_unit_step {
        // Use slice_assign_with_steps
        return slice_assign_with_steps(tensor, indices, value);
    }

    let client = tensor.client.clone();
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };
    let ndims = tensor.shape.num_dims();

    let line_size = if tensor.strides[ndims - 1] == 1 && value.strides[ndims - 1] == 1 {
        let last = indices
            .get(ndims - 1)
            .cloned()
            .unwrap_or(burn_backend::Slice {
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
                let it = **it;
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

    let mut shape = SequenceArg::<R, FastDivmod<usize>>::new();
    let mut offsets = SequenceArg::<R, usize>::new();

    for i in 0..ndims {
        let slice = indices.get(i).cloned().unwrap_or(burn_backend::Slice {
            start: 0,
            end: Some(tensor.shape[i] as isize),
            step: 1,
        });
        let start = slice.start as usize;
        let end = slice.end.unwrap_or(tensor.shape[i] as isize);
        let length = (end - slice.start) as usize;

        shape.push(FastDivmodArgs::<usize>::new(&client, length));
        offsets.push(ScalarArg::new(start));
    }

    let working_units = value.shape.num_elements() / line_size;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        slice_assign_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(line_size),
            linear_view(&value, line_size),
            shape,
            offsets,
            tensor.dtype.into(),
        )
        .expect("Kernel to never fail");
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
pub(crate) fn slice_assign_with_steps<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    slices: &[burn_backend::Slice],
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    // Prepare sequences for kernel
    let mut starts = SequenceArg::<R, usize>::new();
    let mut ends = SequenceArg::<R, usize>::new();
    let mut steps = SequenceArg::<R, i32>::new();

    for (dim, slice) in slices.iter().enumerate() {
        let range = slice.to_range(tensor.shape[dim]);
        starts.push(ScalarArg::new(range.start));
        ends.push(ScalarArg::new(range.end));
        steps.push(ScalarArg::new(slice.step as i32));
    }

    // Pad with default values if needed to match tensor dimensions
    for dim in slices.len()..tensor.shape.num_dims() {
        starts.push(ScalarArg::new(0));
        ends.push(ScalarArg::new(tensor.shape[dim]));
        steps.push(ScalarArg::new(1));
    }

    // Launch kernel
    let working_units = value.shape.num_elements();
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        slice_assign_with_steps_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            linear_view(&value, 1),
            shape_divmod(&value),
            starts,
            ends,
            steps,
            tensor.dtype.into(),
        )
        .expect("Kernel to never fail");
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

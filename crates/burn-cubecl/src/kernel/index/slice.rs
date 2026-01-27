use crate::{
    CubeRuntime,
    kernel::utils::{linear_view, shape_divmod},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_backend::Slice;
use cubecl::{
    calculate_cube_count_elemwise, intrinsic,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearView},
};
use std::ops::Range;

/// Slice a jit tensor with a set of ranges
pub fn slice<R: CubeRuntime>(tensor: CubeTensor<R>, indices: &[Range<usize>]) -> CubeTensor<R> {
    let mut dims = tensor.shape.clone();
    let mut offset_start = 0u64;
    let mut offset_end = 0u64;

    for i in 0..indices.len() {
        offset_start += (tensor.strides[i] * indices[i].start) as u64;
        offset_end += (tensor.strides[i] * (dims[i] - indices[i].end)) as u64;
        dims[i] = indices[i].end - indices[i].start;
    }

    let offset_start = offset_start * tensor.dtype.size() as u64;
    let offset_end = offset_end * tensor.dtype.size() as u64;

    let memory_offset_alignment = tensor.client.properties().memory.alignment;

    if offset_start.is_multiple_of(memory_offset_alignment)
        && offset_end.is_multiple_of(memory_offset_alignment)
    {
        CubeTensor::new(
            tensor.client,
            tensor
                .handle
                .offset_start(offset_start)
                .offset_end(offset_end),
            dims,
            tensor.device,
            tensor.strides,
            tensor.dtype,
        )
    } else {
        let output = empty_device_dtype(
            tensor.client.clone(),
            tensor.device.clone(),
            dims,
            tensor.dtype,
        );
        slice_on_output(tensor, output, indices)
    }
}

#[cube(launch_unchecked)]
fn slice_kernel<E: Numeric>(
    input: &Tensor<E>,
    output: &mut LinearView<E, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    indices: Sequence<usize>,
    #[define(E)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = comptime![out_shape.len()];
    let mut offset_output = ABSOLUTE_POS;
    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        // Iterate in reverse to use divmod
        let dim = rank - i - 1;

        let range_start = indices[dim];
        let (rem, offset_local) = out_shape[dim].div_mod(offset_output);
        offset_output = rem;

        let offset_local = offset_local + range_start;

        offset_input += offset_local * input.stride(dim);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub(crate) fn slice_on_output<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    output: CubeTensor<R>,
    indices: &[Range<usize>],
) -> CubeTensor<R> {
    let ndims = tensor.shape.num_dims();
    let mut indices_sequence = SequenceArg::<R, usize>::new();

    for i in 0..ndims {
        let start = indices.get(i).map(|index| index.start).unwrap_or(0);
        indices_sequence.push(ScalarArg::new(start));
    }

    let working_units = output.shape.num_elements();
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        slice_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            linear_view(&output, 1),
            shape_divmod(&output),
            indices_sequence,
            tensor.dtype.into(),
        )
        .expect("Kernel to never fail");
    };

    output
}

/// Kernel for slicing with steps
#[cube(launch_unchecked)]
fn slice_with_steps_kernel<E: Numeric>(
    input: &Tensor<E>,
    output: &mut LinearView<E, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    starts: Sequence<usize>,
    ends: Sequence<usize>,
    steps: Sequence<i32>,
    #[define(E)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = comptime![out_shape.len()];
    let mut output_offset = ABSOLUTE_POS;
    let mut input_offset = 0;

    // Calculate the input offset based on output position and slice info
    #[unroll]
    for i in 0..rank {
        // Iterate in reverse to use divmod
        let dim = rank - i - 1;
        let start = starts[dim];
        let end = ends[dim];
        let step = steps[dim];

        let (rem, output_idx) = out_shape[dim].div_mod(output_offset);
        output_offset = rem;

        let input_idx = if step > 0 {
            // Forward stepping
            start + output_idx * (step as usize)
        } else {
            // Backward stepping - start from end-1
            let abs_step = (-step) as usize;
            let end_minus_1 = end - 1;
            end_minus_1 - output_idx * abs_step
        };

        input_offset += input_idx * input.stride(dim);
    }

    output[ABSOLUTE_POS] = input[input_offset];
}

/// Slice a tensor with steps
pub fn slice_with_steps<R: CubeRuntime>(tensor: CubeTensor<R>, slices: &[Slice]) -> CubeTensor<R> {
    // Check if all steps are 1 - if so, use the optimized regular slice
    let all_steps_one = slices.iter().all(|info| info.step == 1);

    if all_steps_one {
        // Convert Slice to Range for step=1
        let simple_ranges: Vec<Range<usize>> = slices
            .iter()
            .enumerate()
            .map(|(i, slice)| slice.to_range(tensor.shape[i]))
            .collect();
        return slice(tensor, &simple_ranges);
    }

    // Calculate output shape
    let shape_output = tensor.shape.clone().slice(slices).unwrap();

    // Create output tensor
    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        shape_output.clone(),
        tensor.dtype,
    );

    // Prepare three separate sequences for kernel
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
    let working_units = shape_output.num_elements();
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        slice_with_steps_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            linear_view(&output, 1),
            shape_divmod(&output),
            starts,
            ends,
            steps,
            tensor.dtype.into(),
        )
        .expect("Kernel to never fail");
    }

    output
}

/// This is annoying and we need to find a way to do this automatically at some point
#[allow(unused)]
#[cube]
fn unwrap(value: u32) -> comptime_type!(u32) {
    intrinsic!(|_| value.constant().unwrap().as_u32())
}

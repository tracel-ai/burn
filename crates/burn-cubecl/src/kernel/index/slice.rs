use crate::{CubeRuntime, element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor};
use burn_tensor::{Shape, SliceInfo};
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use std::ops::Range;

/// Slice a jit tensor with a set of ranges
pub fn slice<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    indices: &[Range<usize>],
) -> CubeTensor<R> {
    let mut dims = tensor.shape.dims.clone();
    let mut offset_start = 0u64;
    let mut offset_end = 0u64;

    for i in 0..indices.len() {
        offset_start += (tensor.strides[i] * indices[i].start) as u64;
        offset_end += (tensor.strides[i] * (dims[i] - indices[i].end)) as u64;
        dims[i] = indices[i].end - indices[i].start;
    }

    let offset_start = offset_start * E::cube_type().size() as u64;
    let offset_end = offset_end * E::cube_type().size() as u64;

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
            Shape::from(dims),
            tensor.device,
            tensor.strides,
            tensor.dtype,
        )
    } else {
        let shape_output = Shape::from(dims);
        let output =
            empty_device::<R, E>(tensor.client.clone(), tensor.device.clone(), shape_output);
        slice_on_output::<R, E>(tensor, output, indices)
    }
}

#[cube(launch_unchecked)]
fn slice_kernel<E: CubePrimitive>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    indices: Sequence<u32>,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        let range_start = *indices.index(i);
        let offset_local = ABSOLUTE_POS / output.stride(i) % output.shape(i) + range_start;

        offset_input += offset_local * input.stride(i);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub(crate) fn slice_on_output<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    output: CubeTensor<R>,
    indices: &[Range<usize>],
) -> CubeTensor<R> {
    let ndims = tensor.shape.num_dims();
    let mut indices_sequence = SequenceArg::<R, u32>::new();

    for i in 0..ndims {
        let start = indices.get(i).map(|index| index.start).unwrap_or(0);
        indices_sequence.push(ScalarArg::new(start as u32));
    }

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    unsafe {
        slice_kernel::launch_unchecked::<E, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(1),
            output.as_tensor_arg::<E>(1),
            indices_sequence,
            ndims as u32,
        )
    };

    output
}

/// Kernel for slicing with steps
#[cube(launch_unchecked)]
fn slice_with_steps_kernel<E: CubePrimitive>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    starts: Sequence<u32>,
    ends: Sequence<u32>,
    steps: Sequence<i32>,
    #[comptime] num_dims: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mut input_offset = 0;

    // Calculate the input offset based on output position and slice info
    #[unroll]
    for dim in 0..num_dims {
        let start = *starts.index(dim);
        let end = *ends.index(dim);
        let step = *steps.index(dim);

        let output_idx = (ABSOLUTE_POS / output.stride(dim)) % output.shape(dim);

        let input_idx = if step > 0 {
            // Forward stepping
            start + output_idx * (step as u32)
        } else {
            // Backward stepping - start from end-1
            let abs_step = (-step) as u32;
            let end_minus_1 = end - 1;
            end_minus_1 - output_idx * abs_step
        };

        input_offset += input_idx * input.stride(dim);
    }

    output[ABSOLUTE_POS] = input[input_offset];
}

/// Slice a tensor with steps
pub fn slice_with_steps<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    slice_infos: &[SliceInfo],
) -> CubeTensor<R> {
    // Check if all steps are 1 - if so, use the optimized regular slice
    let all_steps_one = slice_infos.iter().all(|info| info.step == 1);

    if all_steps_one {
        // Convert SliceInfo to Range for step=1
        let simple_ranges: Vec<Range<usize>> =
            slice_infos.iter().map(|info| info.range.clone()).collect();
        return slice::<R, E>(tensor, &simple_ranges);
    }

    // Calculate output shape
    let mut output_dims = tensor.shape.dims.clone();
    for (dim, info) in slice_infos.iter().enumerate() {
        let range_size = info.range.end - info.range.start;
        let step_abs = info.step.unsigned_abs();
        output_dims[dim] = range_size.div_ceil(step_abs);
    }
    let shape_output = Shape::from(output_dims);

    // Create output tensor
    let output = empty_device::<R, E>(
        tensor.client.clone(),
        tensor.device.clone(),
        shape_output.clone(),
    );

    // Prepare three separate sequences for kernel
    let mut starts = SequenceArg::<R, u32>::new();
    let mut ends = SequenceArg::<R, u32>::new();
    let mut steps = SequenceArg::<R, i32>::new();

    for info in slice_infos {
        starts.push(ScalarArg::new(info.range.start as u32));
        ends.push(ScalarArg::new(info.range.end as u32));
        steps.push(ScalarArg::new(info.step as i32));
    }

    // Pad with default values if needed to match tensor dimensions
    for dim in slice_infos.len()..tensor.shape.num_dims() {
        starts.push(ScalarArg::new(0));
        ends.push(ScalarArg::new(tensor.shape.dims[dim] as u32));
        steps.push(ScalarArg::new(1));
    }

    // Launch kernel
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(shape_output.num_elements(), cube_dim);

    unsafe {
        slice_with_steps_kernel::launch_unchecked::<E, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(1),
            output.as_tensor_arg::<E>(1),
            starts,
            ends,
            steps,
            tensor.shape.num_dims() as u32,
        );
    }

    output
}

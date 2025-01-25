use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};
use burn_tensor::Shape;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use std::ops::Range;

pub(crate) fn slice<R: JitRuntime, E: JitElement>(
    tensor: JitTensor<R>,
    indices: &[Range<usize>],
) -> JitTensor<R> {
    let mut dims = tensor.shape.dims.clone();
    let mut offset_start = 0u64;
    let mut offset_end = 0u64;

    for i in 0..indices.len() {
        offset_start += (tensor.strides[i] * indices[i].start) as u64;
        offset_end += (tensor.strides[i] * (dims[i] - indices[i].end)) as u64;
        dims[i] = indices[i].end - indices[i].start;
    }

    let offset_start = offset_start * E::cube_elem().size() as u64;
    let offset_end = offset_end * E::cube_elem().size() as u64;

    let memory_offset_alignment = tensor.client.properties().memory_properties().alignment;

    if offset_start % memory_offset_alignment == 0u64
        && offset_end % memory_offset_alignment == 0u64
    {
        JitTensor::new(
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

pub(crate) fn slice_on_output<R: JitRuntime, E: JitElement>(
    tensor: JitTensor<R>,
    output: JitTensor<R>,
    indices: &[Range<usize>],
) -> JitTensor<R> {
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

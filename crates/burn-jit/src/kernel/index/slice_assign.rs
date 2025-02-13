use crate::{element::CubeElement, tensor::CubeTensor, CubeRuntime};
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use std::ops::Range;

#[cube(launch)]
fn slice_assign_kernel<E: CubePrimitive>(
    input: &mut Tensor<E>,
    value: &Tensor<E>,
    indices: Sequence<u32>,
    #[comptime] rank: u32,
) {
    let mut offset_input = 0;
    let mut offset_value = 0;

    #[unroll]
    for i in 0..rank {
        let range_start = *indices.index(i);
        let offset_local = ABSOLUTE_POS / value.stride(i) % value.shape(i);
        let offset_local_input = offset_local + range_start;

        offset_value += offset_local * value.stride(i);
        offset_input += offset_local_input * input.stride(i);
    }

    input[offset_input] = value[offset_value];
}

pub(crate) fn slice_assign<R: CubeRuntime, E: CubeElement>(
    tensor: CubeTensor<R>,
    indices: &[Range<usize>],
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };
    let ndims = tensor.shape.num_dims();
    let mut indices_sequence = SequenceArg::<R, u32>::new();

    for i in 0..ndims {
        let start = indices.get(i).map(|index| index.start).unwrap_or(0);
        indices_sequence.push(ScalarArg::new(start as u32));
    }

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(tensor.shape.num_elements(), cube_dim);

    slice_assign_kernel::launch::<E, R>(
        &tensor.client,
        cube_count,
        cube_dim,
        tensor.as_tensor_arg::<E>(1),
        value.as_tensor_arg::<E>(1),
        indices_sequence,
        ndims as u32,
    );

    tensor
}

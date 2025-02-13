use crate::{
    element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor, BoolElement, CubeRuntime,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch_unchecked)]
fn flip_kernel<E: CubePrimitive, Bool: Int>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    indices: Sequence<Bool>,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        let stride = input.stride(i);
        let shape = output.shape(i);
        let flip = *indices.index(i) == Bool::from_int(1);
        let mut offset_local = ABSOLUTE_POS / stride % shape;

        if flip {
            offset_local = shape - offset_local - 1;
        }

        offset_input += offset_local * stride;
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub(crate) fn flip<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    tensor: CubeTensor<R>,
    indices: &[usize],
) -> CubeTensor<R> {
    let output = empty_device::<R, E>(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
    );
    flip_on_output::<R, E, BT>(tensor, output, indices)
}

pub(crate) fn flip_on_output<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    tensor: CubeTensor<R>,
    output: CubeTensor<R>,
    indices: &[usize],
) -> CubeTensor<R> {
    let ndims = tensor.shape.num_dims();
    let mut indices_sequence = SequenceArg::<'_, R, BT>::new();

    for i in 0..ndims {
        indices_sequence.push(ScalarArg::new(BT::new_bool(indices.contains(&i))));
    }

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    unsafe {
        flip_kernel::launch_unchecked::<E, BT, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(1),
            output.as_tensor_arg::<E>(1),
            indices_sequence,
            ndims as u32,
        );
    }

    output
}

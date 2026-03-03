use crate::{
    CubeRuntime,
    kernel::utils::{address_type, linear_view, shape_divmod},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_backend::{DType, TensorMetadata};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearView},
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn flip_kernel<E: Numeric, Bool: Int>(
    input: &Tensor<E>,
    output: &mut LinearView<E, ReadWrite>,
    in_shape: Sequence<FastDivmod<usize>>,
    indices: Sequence<InputScalar>,
    #[define(E, Bool)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = in_shape.len().comptime();

    let mut offset = ABSOLUTE_POS;
    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        let dim = rank - i - 1;
        let shape = input.shape(dim);

        let (rem, offset_local) = in_shape[dim].div_mod(offset);
        offset = rem;

        let flip = indices.index(dim).get::<Bool>() == Bool::from_int(1);
        let offset_local = select(flip, shape - offset_local - 1, offset_local);

        offset_input += offset_local * input.stride(dim);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub(crate) fn flip<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    indices: &[usize],
    dtype_bool: DType,
) -> CubeTensor<R> {
    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        tensor.dtype,
    );
    flip_on_output(tensor, output, indices, dtype_bool)
}

pub(crate) fn flip_on_output<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    output: CubeTensor<R>,
    indices: &[usize],
    dtype_bool: DType,
) -> CubeTensor<R> {
    let dtype_input = tensor.dtype;
    let ndims = tensor.meta.num_dims();
    let mut indices_sequence = SequenceArg::<'_, R, InputScalar>::new();

    for i in 0..ndims {
        indices_sequence.push({
            let val = indices.contains(&i) as u8;
            InputScalar::new(val, dtype_bool)
        });
    }

    let num_elements = output.meta.num_elements();
    let cube_dim = CubeDim::new(&tensor.client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, num_elements, cube_dim);

    let shape = shape_divmod(&tensor);
    unsafe {
        flip_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(tensor, output),
            tensor.into_tensor_arg(1),
            linear_view(output.clone(), 1),
            shape,
            indices_sequence,
            [dtype_input.into(), dtype_bool.into()],
        )
    }

    output
}

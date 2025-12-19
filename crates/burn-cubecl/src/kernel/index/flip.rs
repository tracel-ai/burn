use crate::{
    CubeRuntime, kernel::into_contiguous, ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_backend::DType;
use cubecl::std::scalar::InputScalar;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch_unchecked)]
fn flip_kernel<E: Numeric, Bool: Int>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    indices: Sequence<InputScalar>,
    #[comptime] rank: u32,
    #[define(E, Bool)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        let stride = input.stride(i);
        let shape = output.shape(i);
        let flip = indices.index(i).get::<Bool>() == Bool::from_int(1);
        let mut offset_local = ABSOLUTE_POS / stride % shape;

        if flip {
            offset_local = shape - offset_local - 1;
        }

        offset_input += offset_local * stride;
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
        tensor.shape.clone(),
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
    let tensor = into_contiguous(tensor);
    let dtype_input = tensor.dtype;
    let ndims = tensor.shape.num_dims();
    let mut indices_sequence = SequenceArg::<'_, R, InputScalar>::new();

    for i in 0..ndims {
        indices_sequence.push({
            let val = indices.contains(&i) as u8;
            InputScalar::new(val, dtype_bool)
        });
    }

    let num_elements = output.shape.num_elements();
    let cube_dim = CubeDim::new(&tensor.client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, num_elements, cube_dim);

    unsafe {
        flip_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            output.as_tensor_arg(1),
            indices_sequence,
            ndims as u32,
            [dtype_input.into(), dtype_bool.into()],
        )
        .expect("Kernel to never fail");
    }

    output
}

use crate::{
    CubeRuntime, kernel::utils::shape_divmod, ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, FastDivmodArgs},
};

#[cube(launch_unchecked)]
fn repeat_dim_kernel<E: Numeric>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    out_shape: Sequence<FastDivmod<usize>>,
    in_shape: FastDivmod<usize>,
    #[comptime] dim: usize,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let rank = out_shape.len().comptime();

    let mut pos = ABSOLUTE_POS;
    let mut offset_input = 0;
    let mut offset_output = 0;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;

        let (rem, mut local_pos) = out_shape[i].div_mod(pos);
        pos = rem;

        offset_output += local_pos * output.stride(i);

        if i == dim {
            local_pos = in_shape.modulo(local_pos);
        }

        offset_input += local_pos * input.stride(i);
    }

    output[offset_output] = input[offset_input];
}

pub(crate) fn repeat_dim<R: CubeRuntime>(
    mut input: CubeTensor<R>,
    dim: usize,
    times: usize,
) -> CubeTensor<R> {
    if input.shape[dim] == 1 {
        input.strides[dim] = 0;
        input.shape = input.shape.repeat(dim, times).unwrap();
        return input;
    }

    let shape = input.shape.clone().repeat(dim, times).unwrap();

    // Create output handle
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape,
        input.dtype,
    );

    let working_units = output.shape.num_elements();
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    unsafe {
        repeat_dim_kernel::launch_unchecked(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(1),
            output.as_tensor_arg(1),
            shape_divmod(&output),
            FastDivmodArgs::new(&input.client, input.shape[dim]),
            dim,
            output.dtype.into(),
        )
        .expect("Kernel to never fail");
    };

    output
}

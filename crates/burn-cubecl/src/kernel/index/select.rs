use crate::{CubeRuntime, tensor::CubeTensor};
use crate::{
    kernel::utils::{linear_view, shape_divmod},
    ops::numeric::empty_device_dtype,
};
use cubecl::{CubeDim, calculate_cube_count_elemwise, std::tensor::layout::linear::LinearView};
use cubecl::{prelude::*, std::FastDivmod};

#[cube(launch_unchecked)]
fn select_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<T>,
    indices: &LinearView<I>,
    output: &mut LinearView<T, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    dim: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= output.shape() {
        terminate!();
    }

    let rank = out_shape.len().comptime();

    let mut offset = ABSOLUTE_POS;
    let mut offset_input = 0;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        let (rem, offset_local) = out_shape[i].div_mod(offset);
        offset = rem;

        let offset_local = cubecl::prelude::select(
            i == dim,
            usize::cast_from(indices[offset_local]),
            offset_local,
        );

        offset_input += offset_local * input.stride(i);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub(crate) fn select<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    dim: usize,
    indices: CubeTensor<R>,
) -> CubeTensor<R> {
    let mut shape_output = tensor.shape.clone();
    shape_output.dims[dim] = indices.shape[0];
    let total_elem = shape_output.num_elements();

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        shape_output,
        tensor.dtype,
    );

    let working_units = total_elem;
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    unsafe {
        select_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            linear_view(&indices, 1),
            linear_view(&output, 1),
            shape_divmod(&output),
            ScalarArg::new(dim),
            [tensor.dtype.into(), indices.dtype.into()],
        )
        .expect("Kernel to never fail");
    };
    output
}

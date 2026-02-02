use crate::{
    CubeRuntime,
    kernel::utils::{broadcast_strides, linear_view, shape_divmod},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use cubecl::frontend::{ABSOLUTE_POS, Numeric, Tensor};
use cubecl::std::{FastDivmod, tensor::index_offset_contiguous_fastdivmod};
use cubecl::{CubeDim, std::tensor::layout::linear::LinearView};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch_unchecked)]
fn gather_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<Line<T>>,
    indices: &LinearView<Line<I>>,
    output: &mut LinearView<Line<T>, ReadWrite>,
    in_strides: Sequence<usize>, // zeroed out for broadcast dims and `dim`
    out_shape: Sequence<FastDivmod<usize>>,
    dim: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if !indices.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let mut offset = index_offset_contiguous_fastdivmod(
        ABSOLUTE_POS,
        &out_shape,
        &in_strides,
        input.line_size(),
    );

    offset += usize::cast_from(indices[ABSOLUTE_POS]) * input.stride(dim);

    output[ABSOLUTE_POS] = input[offset];
}

pub(crate) fn gather<R: CubeRuntime>(
    dim: usize,
    tensor: CubeTensor<R>,
    indices: CubeTensor<R>,
) -> CubeTensor<R> {
    let shape_output = indices.shape.clone();
    let total_elem = shape_output.num_elements();
    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        shape_output,
        tensor.dtype,
    );

    let cube_dim = CubeDim::new(&tensor.client, total_elem);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, total_elem, cube_dim);
    let mut in_strides = broadcast_strides(&output, &tensor);
    in_strides.values[dim] = ScalarArg::new(0); // Zero `dim` to exclude it from the indexing

    unsafe {
        gather_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            linear_view(&indices, 1),
            linear_view(&output, 1),
            in_strides,
            shape_divmod(&output),
            ScalarArg::new(dim),
            [tensor.dtype.into(), indices.dtype.into()],
        )
        .expect("Kernel to never fail");
    }

    output
}

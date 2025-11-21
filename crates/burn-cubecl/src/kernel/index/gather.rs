use crate::{
    CubeRuntime,
    kernel::utils::{linear_layout, linear_view},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use cubecl::std::tensor::{
    index_offset_with_layout,
    layout::{Layout, LayoutExpand},
};
use cubecl::{CubeDim, std::tensor::layout::linear::LinearView};
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubecl::{
    frontend::{ABSOLUTE_POS, Numeric, Tensor},
    std::tensor::layout::linear::LinearLayout,
};

#[cube(launch_unchecked)]
fn gather_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<Line<T>>,
    indices: &LinearView<Line<I>>,
    output: &mut Tensor<Line<T>>,
    out_layout: LinearLayout,
    dim: &u32,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if !indices.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let index = indices[ABSOLUTE_POS];
    let out_pos = out_layout.to_source_pos(ABSOLUTE_POS);

    let stride = input.stride(*dim);
    let mut offset = u32::cast_from(index);
    offset *= stride;

    if *dim > 0 {
        let offset_before = index_offset_with_layout(input, output, out_pos, 0, *dim, false);
        offset += offset_before;
    }

    let offset_after =
        index_offset_with_layout(input, output, out_pos, *dim + 1, input.rank(), false);
    offset += offset_after;
    output[out_pos] = input[offset];
}

pub(crate) fn gather<R: CubeRuntime>(
    dim: usize,
    tensor: CubeTensor<R>,
    indices: CubeTensor<R>,
) -> CubeTensor<R> {
    let shape_output = indices.shape.clone();
    let total_elem = shape_output.num_elements();
    let output = empty_device_dtype::<R>(
        tensor.client.clone(),
        tensor.device.clone(),
        shape_output,
        tensor.dtype,
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);
    unsafe {
        gather_kernel::launch_unchecked::<R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            linear_view(&indices, 1),
            output.as_tensor_arg(1),
            linear_layout(&output, 1),
            ScalarArg::new(dim as u32),
            [tensor.dtype.into(), indices.dtype.into()],
        )
    }
    output
}

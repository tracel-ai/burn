use crate::{
    CubeRuntime,
    kernel::utils::linear_view,
    ops::{max_line_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_tensor::DType;
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch)]
pub(crate) fn cast_element<I: Numeric, O: Numeric>(
    input: &LinearView<Line<I>>,
    output: &mut LinearView<Line<O>, ReadWrite>,
    #[define(I, O)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Line::cast_from(input[ABSOLUTE_POS]);
}

/// Cast a tensor to the given element type.
///
/// Note: When input element is semantically a boolean, prefer bool_cast function.
pub fn cast<R: CubeRuntime>(input: CubeTensor<R>, dtype: DType) -> CubeTensor<R> {
    let dtype_output = match dtype {
        DType::Flex32 => DType::F32,
        _ => dtype,
    };
    let dtype_input = match input.dtype {
        DType::Flex32 => DType::F32,
        _ => input.dtype,
    };

    if dtype_input == dtype_output {
        return input;
    }

    let line_size = max_line_size(&input);

    let num_elems: usize = input.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);
    let client = input.client.clone();
    let output = empty_device_dtype(
        client.clone(),
        input.device.clone(),
        input.shape.clone(),
        dtype, // We take the same dtype as passed as input (Flex32 not F32)
    );

    cast_element::launch(
        &client,
        cube_count,
        cube_dim,
        linear_view(&input, line_size),
        linear_view(&output, line_size),
        [dtype_input.into(), dtype_output.into()],
    );

    output
}

use crate::{
    CubeElement, CubeRuntime,
    kernel::utils::linear_view,
    ops::{max_line_size, numeric::empty_device},
    tensor::CubeTensor,
};
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use std::any::TypeId;

#[cube(launch)]
pub(crate) fn cast_element<I: CubePrimitive, O: CubePrimitive>(
    input: &LinearView<Line<I>>,
    output: &mut LinearView<Line<O>, ReadWrite>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = Line::cast_from(input[ABSOLUTE_POS]);
}

/// Cast a tensor to the given element type.
///
/// Note: When input element is semantically a boolean, prefer bool_cast function.
pub fn cast<R: CubeRuntime, EI: CubeElement, EO: CubeElement>(
    input: CubeTensor<R>,
) -> CubeTensor<R> {
    if TypeId::of::<EI>() == TypeId::of::<EO>() {
        return CubeTensor::new(
            input.client,
            input.handle,
            input.shape,
            input.device,
            input.strides,
            input.dtype,
        );
    }

    let line_size = max_line_size(&input);

    let num_elems: usize = input.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);
    let client = input.client.clone();
    let output = empty_device::<R, EO>(client.clone(), input.device.clone(), input.shape.clone());

    cast_element::launch::<EI, EO, R>(
        &client,
        cube_count,
        cube_dim,
        linear_view(&input, &line_size),
        linear_view(&output, &line_size),
    );

    output
}

use crate::{
    CubeElement, CubeRuntime, kernel::index_pitched, ops::numeric::empty_device, tensor::CubeTensor,
};
use cubecl::linalg::tensor::index_offset_contiguous;
use cubecl::{calculate_cube_count_elemwise, prelude::*, tensor_vectorization_factor};
use std::any::TypeId;

#[cube(launch)]
pub(crate) fn cast_element<I: CubePrimitive, O: CubePrimitive>(
    input: &Tensor<Line<I>>,
    output: &mut Tensor<Line<O>>,
    #[comptime] rank: Option<u32>,
    #[comptime] pitched: bool,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        terminate!();
    }

    let offset_input = index_offset_contiguous::<I>(input, offset_output, rank);
    let offset_output = index_pitched(output, offset_output, pitched);

    output[offset_output] = Line::cast_from(input[offset_input]);
}

/// Cast a tensor to the given element type.
///
/// Note: When input element is semantically a boolean, prefer bool_cast function.
pub fn cast<R: CubeRuntime, EI: CubeElement, EO: CubeElement>(
    input: CubeTensor<R>,
) -> CubeTensor<R> {
    if TypeId::of::<EI>() == TypeId::of::<EO>() {
        return CubeTensor::new(input.client, input.handle, input.device, input.dtype);
    }

    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = input.shape().num_dims();
    let vectorization_factor = tensor_vectorization_factor(
        R::supported_line_sizes(),
        &input.shape().dims,
        input.strides(),
        rank - 1,
    );

    let num_elems: usize = input.shape().num_elements();

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);
    let client = input.client.clone();
    let output = empty_device::<R, EO>(client.clone(), input.device.clone(), input.shape().clone());

    cast_element::launch::<EI, EO, R>(
        &client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(vectorization_factor),
        output.as_tensor_arg(vectorization_factor),
        Some(rank as u32),
        output.is_pitched(),
    );

    output
}

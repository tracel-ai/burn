use cubecl::{calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*};

use crate::{
    element::CubeElement,
    ops::{max_vectorization, numeric::empty_device},
    tensor::CubeTensor,
    BoolElement, CubeRuntime,
};

#[cube(launch)]
fn mask_where_readonly_kernel<T: CubePrimitive, B: Int>(
    input: &Tensor<Line<T>>,
    mask: &Tensor<Line<B>>,
    value: &Tensor<Line<T>>,
    output: &mut Tensor<Line<T>>,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let index_input = index_offset_with_layout(input, output, ABSOLUTE_POS, 0, rank, true);
    let index_mask = index_offset_with_layout(mask, output, ABSOLUTE_POS, 0, rank, true);
    let index_value = index_offset_with_layout(value, output, ABSOLUTE_POS, 0, rank, true);
    let mask = Line::cast_from(mask[index_mask]);

    output[ABSOLUTE_POS] = select_many(mask, value[index_value], input[index_input]);
}

#[cube(launch)]
fn mask_where_inplace_kernel<T: CubePrimitive, B: Int>(
    input: &mut Tensor<Line<T>>,
    mask: &Tensor<Line<B>>,
    value: &Tensor<Line<T>>,
    reverse: B,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    let index_mask = index_offset_with_layout(mask, input, ABSOLUTE_POS, 0, rank, true);
    let index_value = index_offset_with_layout(value, input, ABSOLUTE_POS, 0, rank, true);

    input[ABSOLUTE_POS] = select(
        mask[index_mask] != Line::new(reverse),
        value[index_value],
        input[ABSOLUTE_POS],
    );
}

#[derive(Clone, Copy, Debug)]
/// Define how to run the mask where kernel.
///
/// # Notes
///
/// All assertions should be done before choosing the strategy.
pub enum MaskWhereStrategy {
    /// Don't mutate any input.
    Readonly,
    /// Reuse the lhs tensor inplace.
    InplaceLhs,
    /// Reuse the rhs tensor inplace.
    InplaceRhs,
}

/// Execute the mask where kernel with the given strategy.
pub fn mask_where<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: CubeTensor<R>,
    strategy: MaskWhereStrategy,
) -> CubeTensor<R> {
    match strategy {
        MaskWhereStrategy::Readonly => mask_where_readonly::<R, E, BT>(input, mask, value),
        MaskWhereStrategy::InplaceLhs => mask_where_inplace::<R, E, BT>(input, mask, value, false),
        MaskWhereStrategy::InplaceRhs => mask_where_inplace::<R, E, BT>(value, mask, input, true),
    }
}

fn mask_where_readonly<R: CubeRuntime, EI: CubeElement, EM: BoolElement>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    let ndims = input.shape.num_dims();
    let output = empty_device::<R, EI>(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);
    let vectorization = max_vectorization(&input);

    mask_where_readonly_kernel::launch::<EI, EM, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<EI>(vectorization),
        mask.as_tensor_arg::<EM>(vectorization),
        value.as_tensor_arg::<EI>(vectorization),
        output.as_tensor_arg::<EI>(vectorization),
        ndims as u32,
    );

    output
}

fn mask_where_inplace<R: CubeRuntime, EI: CubeElement, EM: BoolElement>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: CubeTensor<R>,
    reverse: bool,
) -> CubeTensor<R> {
    let ndims = input.shape.num_dims();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);
    let vectorization = max_vectorization(&input);

    mask_where_inplace_kernel::launch::<EI, EM, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<EI>(vectorization),
        mask.as_tensor_arg::<EM>(vectorization),
        value.as_tensor_arg::<EI>(vectorization),
        ScalarArg::new(EM::new_bool(reverse)),
        ndims as u32,
    );

    input
}

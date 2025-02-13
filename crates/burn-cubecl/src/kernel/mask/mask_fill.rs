use cubecl::{calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*};

use crate::{
    element::CubeElement,
    ops::{max_vectorization, numeric::empty_device},
    tensor::CubeTensor,
    BoolElement, CubeRuntime,
};

#[cube(launch)]
fn mask_fill_readonly_kernel<T: Numeric, B: Int>(
    input: &Tensor<Line<T>>,
    mask: &Tensor<Line<B>>,
    output: &mut Tensor<Line<T>>,
    value: T,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let index_input = index_offset_with_layout(input, output, ABSOLUTE_POS, 0, rank, true);
    let index_mask = index_offset_with_layout(mask, output, ABSOLUTE_POS, 0, rank, true);

    let mask = Line::cast_from(mask[index_mask]);

    output[ABSOLUTE_POS] = select_many(mask, Line::new(value), input[index_input]);
}

#[cube(launch)]
fn mask_fill_inplace_kernel<T: Numeric, B: Int>(
    input: &mut Tensor<Line<T>>,
    mask: &Tensor<Line<B>>,
    value: T,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    let index_mask = index_offset_with_layout(mask, input, ABSOLUTE_POS, 0, rank, true);
    let mask = Line::cast_from(mask[index_mask]);

    input[ABSOLUTE_POS] = select_many(mask, Line::new(value), input[ABSOLUTE_POS]);
}

#[derive(Clone, Copy, Debug)]
/// Define how to run the mask fill kernel.
///
/// # Notes
///
/// All assertions should be done before choosing the strategy.
pub enum MaskFillStrategy {
    /// Don't mutate any input.
    Readonly,
    /// Reuse the input tensor inplace.
    Inplace,
}

/// Execute the mask fill kernel with the given strategy.
pub fn mask_fill<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: E,
    strategy: MaskFillStrategy,
) -> CubeTensor<R> {
    match strategy {
        MaskFillStrategy::Readonly => mask_fill_readonly::<R, E, BT>(input, mask, value),
        MaskFillStrategy::Inplace => mask_fill_inplace::<R, E, BT>(input, mask, value),
    }
}

fn mask_fill_readonly<R: CubeRuntime, EI: CubeElement, EM: BoolElement>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: EI,
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

    mask_fill_readonly_kernel::launch::<EI, EM, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<EI>(vectorization),
        mask.as_tensor_arg::<EM>(vectorization),
        output.as_tensor_arg::<EI>(vectorization),
        ScalarArg::new(value),
        ndims as u32,
    );

    output
}

fn mask_fill_inplace<R: CubeRuntime, EI: CubeElement, EM: BoolElement>(
    input: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: EI,
) -> CubeTensor<R> {
    let ndims = input.shape.num_dims();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);
    let vectorization = max_vectorization(&input);

    mask_fill_inplace_kernel::launch::<EI, EM, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<EI>(vectorization),
        mask.as_tensor_arg::<EM>(vectorization),
        ScalarArg::new(value),
        ndims as u32,
    );

    input
}

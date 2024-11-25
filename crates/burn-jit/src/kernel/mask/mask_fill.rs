use cubecl::{calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*};

use crate::{element::BasicJitElement, ops::empty_device, tensor::JitTensor, JitRuntime};

#[cube(launch)]
fn mask_fill_readonly_kernel<T: Numeric>(
    input: &Tensor<Line<T>>,
    mask: &Tensor<Line<u32>>,
    output: &mut Tensor<Line<T>>,
    value: T,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let index_input = index_offset_with_layout(input, output, ABSOLUTE_POS, 0, rank, true);
    let index_mask = index_offset_with_layout(mask, output, ABSOLUTE_POS, 0, rank, true);

    if mask[index_mask] >= Line::new(1) {
        output[ABSOLUTE_POS] = Line::new(value);
    } else {
        output[ABSOLUTE_POS] = input[index_input];
    }
}

#[cube(launch)]
fn mask_fill_inplace_kernel<T: Numeric>(
    input: &mut Tensor<Line<T>>,
    mask: &Tensor<Line<u32>>,
    value: T,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= input.len() {
        return;
    }

    let index_mask = index_offset_with_layout(mask, input, ABSOLUTE_POS, 0, rank, true);

    if mask[index_mask] >= Line::new(1) {
        input[ABSOLUTE_POS] = Line::new(value);
    }
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
pub fn mask_fill<R: JitRuntime, E: BasicJitElement>(
    input: JitTensor<R>,
    mask: JitTensor<R>,
    value: E,
    strategy: MaskFillStrategy,
) -> JitTensor<R> {
    match strategy {
        MaskFillStrategy::Readonly => mask_fill_readonly::<R, E, u32>(input, mask, value),
        MaskFillStrategy::Inplace => mask_fill_inplace::<R, E, u32>(input, mask, value),
    }
}

fn mask_fill_readonly<R: JitRuntime, EI: BasicJitElement, EM: BasicJitElement>(
    input: JitTensor<R>,
    mask: JitTensor<R>,
    value: EI,
) -> JitTensor<R> {
    let ndims = input.shape.num_dims();
    let output = empty_device::<R, EI>(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    mask_fill_readonly_kernel::launch::<EI, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<EI>(1),
        mask.as_tensor_arg::<EM>(1),
        output.as_tensor_arg::<EI>(1),
        ScalarArg::new(value),
        ndims as u32,
    );

    output
}

fn mask_fill_inplace<R: JitRuntime, EI: BasicJitElement, EM: BasicJitElement>(
    input: JitTensor<R>,
    mask: JitTensor<R>,
    value: EI,
) -> JitTensor<R> {
    let ndims = input.shape.num_dims();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    mask_fill_inplace_kernel::launch::<EI, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<EI>(1),
        mask.as_tensor_arg::<EM>(1),
        ScalarArg::new(value),
        ndims as u32,
    );

    input
}

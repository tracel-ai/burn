use cubecl::{calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*};

use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};

#[cube(launch)]
fn mask_where_readonly_kernel<T: CubePrimitive>(
    input: &Tensor<T>,
    mask: &Tensor<u32>,
    value: &Tensor<T>,
    output: &mut Tensor<T>,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let index_input = index_offset_with_layout(input, output, ABSOLUTE_POS, 0, rank, true);
    let index_mask = index_offset_with_layout(mask, output, ABSOLUTE_POS, 0, rank, true);
    let index_value = index_offset_with_layout(value, output, ABSOLUTE_POS, 0, rank, true);

    if mask[index_mask] >= 1 {
        output[ABSOLUTE_POS] = value[index_value];
    } else {
        output[ABSOLUTE_POS] = input[index_input];
    }
}

#[cube(launch)]
fn mask_where_inplace_kernel<T: CubePrimitive>(
    input: &mut Tensor<T>,
    mask: &Tensor<u32>,
    value: &Tensor<T>,
    reverse: u32,
    #[comptime] rank: u32,
) {
    if ABSOLUTE_POS >= input.len() {
        return;
    }

    let index_mask = index_offset_with_layout(mask, input, ABSOLUTE_POS, 0, rank, true);
    let index_value = index_offset_with_layout(value, input, ABSOLUTE_POS, 0, rank, true);

    if mask[index_mask] != reverse {
        input[ABSOLUTE_POS] = value[index_value];
    }
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
pub fn mask_where<R: JitRuntime, E: JitElement>(
    input: JitTensor<R, E>,
    mask: JitTensor<R, u32>,
    value: JitTensor<R, E>,
    strategy: MaskWhereStrategy,
) -> JitTensor<R, E> {
    match strategy {
        MaskWhereStrategy::Readonly => mask_where_readonly(input, mask, value),
        MaskWhereStrategy::InplaceLhs => mask_where_inplace(input, mask, value, false),
        MaskWhereStrategy::InplaceRhs => mask_where_inplace(value, mask, input, true),
    }
}

fn mask_where_readonly<R: JitRuntime, EI: JitElement, EM: JitElement>(
    input: JitTensor<R, EI>,
    mask: JitTensor<R, EM>,
    value: JitTensor<R, EI>,
) -> JitTensor<R, EI> {
    let ndims = input.shape.num_dims();
    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    mask_where_readonly_kernel::launch::<EI, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        mask.as_tensor_arg(1),
        value.as_tensor_arg(1),
        output.as_tensor_arg(1),
        ndims as u32,
    );

    output
}

fn mask_where_inplace<R: JitRuntime, EI: JitElement, EM: JitElement>(
    input: JitTensor<R, EI>,
    mask: JitTensor<R, EM>,
    value: JitTensor<R, EI>,
    reverse: bool,
) -> JitTensor<R, EI> {
    let ndims = input.shape.num_dims();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    mask_where_inplace_kernel::launch::<EI, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        mask.as_tensor_arg(1),
        value.as_tensor_arg(1),
        ScalarArg::new(reverse as u32),
        ndims as u32,
    );

    input
}

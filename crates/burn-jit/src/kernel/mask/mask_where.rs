use cubecl::{calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*};

use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};

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
pub fn mask_where<R: JitRuntime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    mask: JitTensor<R, u32, D>,
    value: JitTensor<R, E, D>,
    strategy: MaskWhereStrategy,
) -> JitTensor<R, E, D> {
    match strategy {
        MaskWhereStrategy::Readonly => mask_where_readonly(input, mask, value),
        MaskWhereStrategy::InplaceLhs => mask_where_inplace(input, mask, value, false),
        MaskWhereStrategy::InplaceRhs => mask_where_inplace(value, mask, input, true),
    }
}

#[cube(launch)]
fn mask_where_readonly_kernel<T: CubePrimitive>(
    input: &Tensor<T>,
    mask: &Tensor<Bool>,
    value: &Tensor<T>,
    output: &mut Tensor<T>,
    rank: Comptime<Option<UInt>>,
) {
    let index_input = index_offset_with_layout(
        input,
        output,
        ABSOLUTE_POS,
        UInt::new(0),
        Comptime::unwrap_or_else(rank, || output.rank()),
        Comptime::is_some(rank),
    );

    let index_mask = index_offset_with_layout(
        mask,
        output,
        ABSOLUTE_POS,
        UInt::new(0),
        Comptime::unwrap_or_else(rank, || output.rank()),
        Comptime::is_some(rank),
    );

    let index_value = index_offset_with_layout(
        value,
        output,
        ABSOLUTE_POS,
        UInt::new(0),
        Comptime::unwrap_or_else(rank, || output.rank()),
        Comptime::is_some(rank),
    );

    let masked = mask[index_mask];

    if masked {
        output[ABSOLUTE_POS] = value[index_value];
    } else {
        output[ABSOLUTE_POS] = input[index_input];
    }
}

#[cube(launch)]
fn mask_where_inplace_kernel<T: CubePrimitive>(
    input: &mut Tensor<T>,
    mask: &Tensor<Bool>,
    value: &Tensor<T>,
    reverse: UInt,
    rank: Comptime<Option<UInt>>,
) {
    let index_mask = index_offset_with_layout(
        mask,
        input,
        ABSOLUTE_POS,
        UInt::new(0),
        Comptime::unwrap_or_else(rank, || input.rank()),
        Comptime::is_some(rank),
    );

    let index_value = index_offset_with_layout(
        value,
        input,
        ABSOLUTE_POS,
        UInt::new(0),
        Comptime::unwrap_or_else(rank, || input.rank()),
        Comptime::is_some(rank),
    );

    let masked = mask[index_mask] != Bool::cast_from(reverse);

    if masked {
        input[ABSOLUTE_POS] = value[index_value];
    }
}

fn mask_where_readonly<R: JitRuntime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: JitTensor<R, EI, D>,
) -> JitTensor<R, EI, D> {
    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    mask_where_readonly_kernel::launch::<EI::Primitive, R>(
        &input.client,
        cube_count,
        cube_dim,
        TensorArg::new(&input.handle, &input.strides, &input.shape.dims),
        TensorArg::new(&mask.handle, &mask.strides, &mask.shape.dims),
        TensorArg::new(&value.handle, &value.strides, &value.shape.dims),
        TensorArg::new(&output.handle, &output.strides, &output.shape.dims),
        Some(UInt::new(D as u32)),
    );

    output
}

fn mask_where_inplace<R: JitRuntime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: JitTensor<R, EI, D>,
    reverse: bool,
) -> JitTensor<R, EI, D> {
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    mask_where_inplace_kernel::launch::<EI::Primitive, R>(
        &input.client,
        cube_count,
        cube_dim,
        TensorArg::new(&input.handle, &input.strides, &input.shape.dims),
        TensorArg::new(&mask.handle, &mask.strides, &mask.shape.dims),
        TensorArg::new(&value.handle, &value.strides, &value.shape.dims),
        ScalarArg::new(reverse as u32),
        Some(UInt::new(D as u32)),
    );

    input
}

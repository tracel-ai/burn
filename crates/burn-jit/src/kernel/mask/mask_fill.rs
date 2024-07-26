use cubecl::{calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*};

use crate::{
    element::JitElement,
    ops::numeric::empty_device,
    tensor::{self, JitTensor},
    JitRuntime,
};

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
pub fn mask_fill<R: JitRuntime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    mask: JitTensor<R, u32, D>,
    value: E,
    strategy: MaskFillStrategy,
) -> JitTensor<R, E, D> {
    match strategy {
        MaskFillStrategy::Readonly => mask_fill_readonly(input, mask, value),
        MaskFillStrategy::Inplace => mask_fill_inplace(input, mask, value),
    }
}

#[cube(launch)]
fn mask_fill_readonly_kernel<T: CubePrimitive>(
    input: &Tensor<T>,
    mask: &Tensor<Bool>,
    output: &mut Tensor<T>,
    value: T,
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

    let masked = mask[index_mask];

    if masked {
        output[ABSOLUTE_POS] = value;
    } else {
        output[ABSOLUTE_POS] = input[index_input];
    }
}

#[cube(launch)]
fn mask_fill_inplace_kernel<T: CubePrimitive>(
    input: &mut Tensor<T>,
    mask: &Tensor<Bool>,
    value: T,
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

    let masked = mask[index_mask];

    if masked {
        input[ABSOLUTE_POS] = value;
    }
}

fn mask_fill_readonly<R: JitRuntime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: EI,
) -> JitTensor<R, EI, D> {
    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    mask_fill_readonly_kernel::launch::<EI::Primitive, R>(
        &input.client,
        cube_count,
        cube_dim,
        TensorArg::new(&input.handle, &input.strides, &input.shape.dims),
        TensorArg::new(&mask.handle, &mask.strides, &mask.shape.dims),
        TensorArg::new(&output.handle, &output.strides, &output.shape.dims),
        ScalarArg::new(value),
        Some(UInt::new(D as u32)),
    );

    output
}

fn mask_fill_inplace<R: JitRuntime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: EI,
) -> JitTensor<R, EI, D> {
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(input.shape.num_elements(), cube_dim);

    mask_fill_inplace_kernel::launch::<EI::Primitive, R>(
        &input.client,
        cube_count,
        cube_dim,
        TensorArg::new(&input.handle, &input.strides, &input.shape.dims),
        TensorArg::new(&mask.handle, &mask.strides, &mask.shape.dims),
        ScalarArg::new(value),
        Some(UInt::new(D as u32)),
    );

    input
}

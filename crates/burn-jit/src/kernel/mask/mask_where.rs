use burn_cube::{frontend::TensorHandle, CubeCountSettings, Execution};

use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};

use super::{MaskInplaceEagerKernel, MaskReadOnlyEagerKernel, MaskWhere};

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

fn mask_where_readonly<R: JitRuntime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: JitTensor<R, EI, D>,
) -> JitTensor<R, EI, D> {
    let client = input.client.clone();
    let kernel = MaskReadOnlyEagerKernel::<MaskWhere, R, EI, EM>::new(false);

    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    Execution::start(kernel, client)
        .inputs(&[
            TensorHandle::<R>::new(&input.handle, &input.strides, &input.shape.dims),
            TensorHandle::new(&mask.handle, &mask.strides, &mask.shape.dims),
            TensorHandle::new(&value.handle, &value.strides, &value.shape.dims),
        ])
        .outputs(&[TensorHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}

fn mask_where_inplace<R: JitRuntime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: JitTensor<R, EI, D>,
    reverse: bool,
) -> JitTensor<R, EI, D> {
    let kernel = MaskInplaceEagerKernel::<MaskWhere, R, EI, EM>::new(reverse);

    let client = input.client.clone();

    Execution::start(kernel, client)
        .inputs(&[
            TensorHandle::<R>::new(&input.handle, &input.strides, &input.shape.dims),
            TensorHandle::new(&mask.handle, &mask.strides, &mask.shape.dims),
            TensorHandle::new(&value.handle, &value.strides, &value.shape.dims),
        ])
        .execute(CubeCountSettings::Input { pos: 0 });

    input
}

use crate::{
    codegen::{EagerHandle, Execution, WorkgroupLaunch},
    element::JitElement,
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};

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
pub fn mask_where<R: Runtime, E: JitElement, const D: usize>(
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

fn mask_where_readonly<R: Runtime, EI: JitElement, EM: JitElement, const D: usize>(
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
            EagerHandle::<R>::new(&input.handle, &input.strides, &input.shape.dims),
            EagerHandle::new(&mask.handle, &mask.strides, &mask.shape.dims),
            EagerHandle::new(&value.handle, &value.strides, &value.shape.dims),
        ])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}

fn mask_where_inplace<R: Runtime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: JitTensor<R, EI, D>,
    reverse: bool,
) -> JitTensor<R, EI, D> {
    let kernel = MaskInplaceEagerKernel::<MaskWhere, R, EI, EM>::new(reverse);

    let client = input.client.clone();

    Execution::start(kernel, client)
        .inputs(&[
            EagerHandle::<R>::new(&input.handle, &input.strides, &input.shape.dims),
            EagerHandle::new(&mask.handle, &mask.strides, &mask.shape.dims),
            EagerHandle::new(&value.handle, &value.strides, &value.shape.dims),
        ])
        .execute(WorkgroupLaunch::Input { pos: 0 });

    input
}

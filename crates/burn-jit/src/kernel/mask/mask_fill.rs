use burn_cube::{frontend::TensorHandle, CubeCountSettings, Execution};

use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};

use super::{MaskFill, MaskInplaceEagerKernel, MaskReadOnlyEagerKernel};

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

fn mask_fill_readonly<R: JitRuntime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: EI,
) -> JitTensor<R, EI, D> {
    let client = input.client.clone();
    let kernel = MaskReadOnlyEagerKernel::<MaskFill, R, EI, EM>::new(false);

    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    Execution::start(kernel, client)
        .inputs(&[
            TensorHandle::<R>::new(&input.handle, &input.strides, &input.shape.dims),
            TensorHandle::new(&mask.handle, &mask.strides, &mask.shape.dims),
        ])
        .outputs(&[TensorHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .with_scalars(&[value])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}

fn mask_fill_inplace<R: JitRuntime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: EI,
) -> JitTensor<R, EI, D> {
    let kernel = MaskInplaceEagerKernel::<MaskFill, R, EI, EM>::new(false);

    let client = input.client.clone();

    Execution::start(kernel, client)
        .inputs(&[
            TensorHandle::<R>::new(&input.handle, &input.strides, &input.shape.dims),
            TensorHandle::new(&mask.handle, &mask.strides, &mask.shape.dims),
        ])
        .with_scalars(&[value])
        .execute(CubeCountSettings::Input { pos: 0 });

    input
}

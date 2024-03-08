use crate::{
    codegen::{execute_dynamic, EagerHandle, WorkgroupLaunch},
    compute::StaticKernel,
    element::JitElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};

use super::{MaskFill, MaskReadOnlyEagerKernel};

// kernel_wgsl!(MaskFill, "../../template/mask/fill.wgsl");
// kernel_wgsl!(MaskFillInplace, "../../template/mask/fill_inplace.wgsl");

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
pub fn mask_fill<R: Runtime, E: JitElement, const D: usize>(
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

fn mask_fill_readonly<R: Runtime, EI: JitElement, EM: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
    mask: JitTensor<R, EM, D>,
    value: EI,
) -> JitTensor<R, EI, D> {
    let kernel = MaskReadOnlyEagerKernel::new(false);

    let num_elems = input.shape.num_elements();
    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    execute_dynamic::<R, MaskReadOnlyEagerKernel<MaskFill<EI>, R, EI, EM>, EI>(
        &[
            EagerHandle::new(&input.handle, &input.strides, &input.shape.dims),
            EagerHandle::new(&mask.handle, &mask.strides, &mask.shape.dims),
        ],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        Some(&[value]),
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        input.client,
    );

    output
}

fn mask_fill_inplace<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    mask: JitTensor<R, u32, D>,
    value: E,
) -> JitTensor<R, E, D> {
    let num_elems = input.shape.num_elements();
    let value_handle = input.client.create(E::as_bytes(&[value]));
    let kernel = StaticKernel::<
        KernelSettings<MaskFillInplace, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));
    let mask = JitTensor::new(mask.client, mask.device, mask.shape, mask.handle);
    let info = build_info(&[&input, &mask]);
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    input.client.execute(
        Box::new(kernel),
        &[&input.handle, &value_handle, &mask.handle, &info_handle],
    );

    input
}

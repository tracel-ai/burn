use crate::{
    compute::StaticKernel,
    element::JitElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};

kernel_wgsl!(MaskWhere, "../../template/mask/where.wgsl");
kernel_wgsl!(MaskWhereInplace, "../../template/mask/where_inplace.wgsl");

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

fn mask_where_readonly<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    mask: JitTensor<R, u32, D>,
    value: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let num_elems = input.shape.num_elements();
    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    let kernel = StaticKernel::<
        KernelSettings<MaskWhere, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));
    let mask = JitTensor::new(mask.client, mask.device, mask.shape, mask.handle);
    let info = build_info(&[&input, &value, &mask, &output]);
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    input.client.execute(
        Box::new(kernel),
        &[
            &input.handle,
            &value.handle,
            &mask.handle,
            &output.handle,
            &info_handle,
        ],
    );

    output
}

fn mask_where_inplace<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    mask: JitTensor<R, u32, D>,
    value: JitTensor<R, E, D>,
    reverse: bool,
) -> JitTensor<R, E, D> {
    let kernel = StaticKernel::<
        KernelSettings<MaskWhereInplace, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        input.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));
    let mask = JitTensor::new(mask.client, mask.device, mask.shape, mask.handle);
    let mut info = build_info(&[&input, &value, &mask]);
    info.push(match reverse {
        true => 1,
        false => 0,
    });
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    input.client.execute(
        Box::new(kernel),
        &[&input.handle, &value.handle, &mask.handle, &info_handle],
    );

    input
}

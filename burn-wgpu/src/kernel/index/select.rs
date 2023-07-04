use crate::{
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(IndexSelect, "../../template/index/index_select.wgsl");
kernel_wgsl!(
    IndexSelectAssignInplace,
    "../../template/index/index_select_assign_inplace.wgsl"
);

pub(crate) fn select<E: WgpuElement, I: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    dim: usize,
    indexes: WgpuTensor<I, 1>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let mut output_shape = tensor.shape.clone();
    output_shape.dims[dim] = indexes.shape.dims[0];
    let num_elems = output_shape.num_elements();

    let buffer = tensor
        .context
        .create_buffer(num_elems * std::mem::size_of::<E>());
    let output = WgpuTensor::new(tensor.context.clone(), output_shape, buffer);

    let mut info = build_info(&[&tensor, &output]);
    info.push(dim as u32);

    let info_buffer = tensor
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = tensor
        .context
        .compile_static::<KernelSettings<IndexSelect, E, I, WORKGROUP, WORKGROUP, 1>>();

    tensor.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[
            &tensor.buffer,
            &indexes.buffer,
            &output.buffer,
            &info_buffer,
        ],
    );

    output
}

pub(crate) fn select_assign<E: WgpuElement, I: WgpuElement, const D: usize, const D2: usize>(
    tensor: WgpuTensor<E, D>,
    dim: usize,
    indexes: WgpuTensor<I, 1>,
    values: WgpuTensor<E, D2>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let mut shape = tensor.shape.clone();
    shape.dims[dim] = values.shape.dims[dim];
    let values = WgpuTensor::new(values.context, shape, values.buffer);
    let mut info = build_info(&[&tensor, &values]);
    info.push(dim as u32);

    let info_buffer = tensor
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = tensor.context.compile_static::<KernelSettings<
        IndexSelectAssignInplace,
        E,
        I,
        WORKGROUP,
        WORKGROUP,
        1,
    >>();

    let mut shape_tmp = values.shape;
    shape_tmp.dims[dim] = 1; // Just one thread for the dim.

    tensor.context.execute(
        elemwise_workgroup(shape_tmp.num_elements(), WORKGROUP),
        kernel,
        &[
            &tensor.buffer,
            &indexes.buffer,
            &values.buffer,
            &info_buffer,
        ],
    );

    tensor
}

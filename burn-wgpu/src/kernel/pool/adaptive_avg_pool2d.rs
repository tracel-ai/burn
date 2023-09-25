use crate::{
    compute::{StaticKernel, WgpuHandle},
    element::WgpuElement,
    kernel::{elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};
use burn_tensor::Shape;

kernel_wgsl!(
    AdaptiveAvgPool2d,
    "../../template/pool/adaptive_avg_pool2d.wgsl"
);
kernel_wgsl!(
    AdaptiveAvgPool2dBackward,
    "../../template/pool/adaptive_avg_pool2d_backward.wgsl"
);

pub(crate) fn adaptive_avg_pool2d<E: WgpuElement>(
    x: WgpuTensor<E, 4>,
    output_size: [usize; 2],
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let [batch_size, channels, _, _] = x.shape.dims;

    let output_shape = Shape::new([batch_size, channels, output_size[0], output_size[1]]);
    let output = empty_device(x.client.clone(), x.device.clone(), output_shape);

    let kernel =
        StaticKernel::<KernelSettings<AdaptiveAvgPool2d, E, i32, WORKGROUP, WORKGROUP, 1>>::new(
            elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        );

    let info_handle = build_info(&x, &output);
    x.client
        .execute(Box::new(kernel), &[&x.handle, &output.handle, &info_handle]);

    output
}

pub(crate) fn adaptive_avg_pool2d_backward<E: WgpuElement>(
    x: WgpuTensor<E, 4>,
    out_grad: WgpuTensor<E, 4>,
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let output_shape = x.shape.clone();
    let num_elems = output_shape.num_elements();
    let output_buffer = x.client.empty(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(
        x.client.clone(),
        x.device.clone(),
        output_shape,
        output_buffer,
    );

    let kernel = StaticKernel::<
        KernelSettings<AdaptiveAvgPool2dBackward, E, i32, WORKGROUP, WORKGROUP, 1>,
    >::new(elemwise_workgroup(output.shape.num_elements(), WORKGROUP));

    let info_handle = build_info(&x, &out_grad);

    x.client.execute(
        Box::new(kernel),
        &[&out_grad.handle, &output.handle, &info_handle],
    );

    output
}

fn build_info<E: WgpuElement>(x: &WgpuTensor<E, 4>, output: &WgpuTensor<E, 4>) -> WgpuHandle {
    let mut info: [u32; 16] = [0; 16];
    info[0] = x.strides[0] as u32;
    info[1] = x.strides[1] as u32;
    info[2] = x.strides[2] as u32;
    info[3] = x.strides[3] as u32;
    info[4] = x.shape.dims[0] as u32;
    info[5] = x.shape.dims[1] as u32;
    info[6] = x.shape.dims[2] as u32;
    info[7] = x.shape.dims[3] as u32;

    info[8] = output.strides[0] as u32;
    info[9] = output.strides[1] as u32;
    info[10] = output.strides[2] as u32;
    info[11] = output.strides[3] as u32;
    info[12] = output.shape.dims[0] as u32;
    info[13] = output.shape.dims[1] as u32;
    info[14] = output.shape.dims[2] as u32;
    info[15] = output.shape.dims[3] as u32;

    output.client.create(bytemuck::cast_slice(&info))
}

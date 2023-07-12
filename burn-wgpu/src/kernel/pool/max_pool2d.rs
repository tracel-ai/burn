use burn_tensor::Shape;

use crate::{
    element::WgpuElement,
    kernel::{elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(MaxPool2d, "../../template/pool/max_pool2d.wgsl");

pub(crate) fn max_pool2d<E: WgpuElement>(
    x: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 16;

    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [batch_size, channels, x_height, x_width] = x.shape.dims;

    let out_height = ((x_height + 2 * padding_height - kernel_height) / stride_height) + 1;
    let out_width = ((x_width + 2 * padding_width - kernel_width) / stride_width) + 1;
    let shape_out = Shape::new([batch_size, channels, out_height, out_width]);
    let num_elems = shape_out.num_elements();

    let buffer = x
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(x.context.clone(), shape_out, buffer);

    let mut info: [u32; 22] = [0; 22];
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

    info[16] = kernel_height as u32;
    info[17] = kernel_width as u32;
    info[18] = stride_height as u32;
    info[19] = stride_width as u32;
    info[20] = padding_height as u32;
    info[21] = padding_width as u32;

    let info_buffers = x
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = x
        .context
        .compile_static::<KernelSettings<MaxPool2d, E, i32, WORKGROUP, WORKGROUP, 1>>();

    x.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&x.buffer, &output.buffer, &info_buffers],
    );

    output
}

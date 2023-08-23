use crate::{element::WgpuElement, tensor::WgpuTensor};
use burn_tensor::Shape;
use std::sync::Arc;
use wgpu::Buffer;

/// Build basic info to launch pool 2d kernels.
pub fn build_output_and_info_pool2d<E: WgpuElement>(
    x: &WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (Arc<Buffer>, WgpuTensor<E, 4>) {
    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [dilation_height, dilation_width] = dilation;
    let [batch_size, channels, x_height, x_width] = x.shape.dims;

    let out_height = ((x_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1)
        / stride_height)
        + 1;
    let out_width = ((x_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1)
        / stride_width)
        + 1;
    let shape_out = Shape::new([batch_size, channels, out_height, out_width]);
    let num_elems = shape_out.num_elements();

    let buffer = x
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(x.context.clone(), shape_out, buffer);

    let info_buffer = build_pool2d_info(x, &output, kernel_size, stride, padding, dilation);

    (info_buffer, output)
}

pub fn build_pool2d_info<E: WgpuElement>(
    input: &WgpuTensor<E, 4>,
    output: &WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Arc<Buffer> {
    let mut info: [u32; 24] = [0; 24];
    info[0] = input.strides[0] as u32;
    info[1] = input.strides[1] as u32;
    info[2] = input.strides[2] as u32;
    info[3] = input.strides[3] as u32;
    info[4] = input.shape.dims[0] as u32;
    info[5] = input.shape.dims[1] as u32;
    info[6] = input.shape.dims[2] as u32;
    info[7] = input.shape.dims[3] as u32;

    info[8] = output.strides[0] as u32;
    info[9] = output.strides[1] as u32;
    info[10] = output.strides[2] as u32;
    info[11] = output.strides[3] as u32;
    info[12] = output.shape.dims[0] as u32;
    info[13] = output.shape.dims[1] as u32;
    info[14] = output.shape.dims[2] as u32;
    info[15] = output.shape.dims[3] as u32;

    info[16] = kernel_size[0] as u32;
    info[17] = kernel_size[1] as u32;
    info[18] = stride[0] as u32;
    info[19] = stride[1] as u32;
    info[20] = padding[0] as u32;
    info[21] = padding[1] as u32;
    info[22] = dilation[0] as u32;
    info[23] = dilation[1] as u32;

    let info_buffer = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    info_buffer
}

use crate::{element::JitElement, tensor::JitTensor, Runtime};
use burn_compute::server::Handle;

pub fn build_pool2d_info<R: Runtime, E: JitElement>(
    input: &JitTensor<R, E, 4>,
    output: &JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Handle<R::Server> {
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

    let info_buffer = input.client.create(bytemuck::cast_slice(&info));

    info_buffer
}

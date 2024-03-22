use crate::{
    compute::{Kernel, StaticKernel},
    kernel::{self, elemwise_workgroup, KernelSettings, StaticKernelSource, WORKGROUP_DEFAULT},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::JitTensor,
    JitElement, Runtime,
};
use burn_compute::server::Handle;

kernel_wgsl!(
    AvgPool2dBackwardRaw,
    "../../template/pool/avg_pool2d_backward.wgsl"
);

struct AvgPool2dBackward<const COUNT_INCLUDE_PAD: bool>;

impl<const COUNT_INCLUDE_PAD: bool> StaticKernelSource for AvgPool2dBackward<COUNT_INCLUDE_PAD> {
    fn source() -> kernel::SourceTemplate {
        AvgPool2dBackwardRaw::source().register("count_include_pad", format!("{COUNT_INCLUDE_PAD}"))
    }
}

pub(crate) fn avg_pool2d_backward<R: Runtime, E: JitElement>(
    x: JitTensor<R, E, 4>,
    grad: JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> JitTensor<R, E, 4> {
    let grad = kernel::into_contiguous(grad);
    let output = empty_device(x.client.clone(), x.device.clone(), x.shape.clone());
    let info_handle = build_pool2d_info(&x, &grad, kernel_size, stride, padding, [1, 1]);
    let workgroup = elemwise_workgroup(output.shape.num_elements(), WORKGROUP_DEFAULT);

    let kernel: Box<dyn Kernel> = match count_include_pad {
        true => Box::new(StaticKernel::<
            KernelSettings<
                AvgPool2dBackward<true>,
                E,
                i32,
                WORKGROUP_DEFAULT,
                WORKGROUP_DEFAULT,
                1,
            >,
        >::new(workgroup)),
        false => Box::new(StaticKernel::<
            KernelSettings<
                AvgPool2dBackward<false>,
                E,
                i32,
                WORKGROUP_DEFAULT,
                WORKGROUP_DEFAULT,
                1,
            >,
        >::new(workgroup)),
    };

    x.client
        .execute(kernel, &[&grad.handle, &output.handle, &info_handle]);

    output
}

pub(crate) fn build_pool2d_info<R: Runtime, E: JitElement>(
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

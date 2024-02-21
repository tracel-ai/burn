use crate::{
    compute::{Kernel, StaticKernel},
    element::WgpuElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};
use burn_tensor::{
    ops::{InterpolateMode, InterpolateOptions},
    Element, Shape,
};

kernel_wgsl!(Nearest, "../template/interpolate/nearest.wgsl");
kernel_wgsl!(Bilinear, "../template/interpolate/bilinear.wgsl");

pub(crate) fn interpolate<E: WgpuElement + Element>(
    input: WgpuTensor<E, 4>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> WgpuTensor<E, 4> {
    let input = kernel::into_contiguous(input);
    let [batch_size, channels, _, _] = input.shape.dims;
    let [out_height, out_width] = output_size;

    let shape_out = Shape::new([batch_size, channels, out_height, out_width]);
    let output = empty_device(input.client.clone(), input.device.clone(), shape_out);

    let info = build_info::<E, 4>(&[&input, &output]);

    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    let kernel: Box<dyn Kernel> = match options.mode {
        InterpolateMode::Nearest => Box::new(StaticKernel::<
            KernelSettings<Nearest, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
        >::new(elemwise_workgroup(
            output.shape.num_elements(),
            WORKGROUP_DEFAULT,
        ))),
        InterpolateMode::Bilinear => Box::new(StaticKernel::<
            KernelSettings<Bilinear, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
        >::new(elemwise_workgroup(
            output.shape.num_elements(),
            WORKGROUP_DEFAULT,
        ))),
        InterpolateMode::Bicubic => {
            todo!()
        }
    };

    input
        .client
        .execute(kernel, &[&input.handle, &output.handle, &info_handle]);

    output
}

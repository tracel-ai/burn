use crate::{
    compute::StaticKernel,
    element::JitElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::{ops::ConvTransposeOptions, Element, ElementConversion, Shape};

kernel_wgsl!(ConvTranspose2d, "../../template/conv/conv_transpose2d.wgsl");

pub(crate) fn conv_transpose2d<R: Runtime, E: JitElement + Element>(
    input: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvTransposeOptions<2>,
) -> JitTensor<R, E, 4> {
    let input = kernel::into_contiguous(input);
    let weight = kernel::into_contiguous(weight);
    let [batch_size, _, in_height, in_width] = input.shape.dims;
    let [_, out_channels, kernel_0, kernel_1] = weight.shape.dims;

    let out_0 = (in_height - 1) * options.stride[0]
        + options.dilation[0] * (kernel_0 - 1)
        + options.padding_out[0]
        - 2 * options.padding[0]
        + 1;
    let out_1 = (in_width - 1) * options.stride[1]
        + options.dilation[1] * (kernel_1 - 1)
        + options.padding_out[1]
        - 2 * options.padding[1]
        + 1;

    let shape_out = Shape::new([batch_size, out_channels * options.groups, out_0, out_1]);
    let num_elems = shape_out.num_elements();

    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
    );
    let mut info = build_info(&[&input, &output, &weight]);

    info.push(options.stride[0] as u32);
    info.push(options.stride[1] as u32);
    info.push(options.padding[0] as u32);
    info.push(options.padding[1] as u32);
    info.push(options.dilation[0] as u32);
    info.push(options.dilation[1] as u32);
    info.push(options.groups as u32);

    let bias_handle = bias
        .map(|bias| bias.handle)
        .unwrap_or_else(|| input.client.create(E::as_bytes(&[0.elem()])));

    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    let kernel = StaticKernel::<
        KernelSettings<ConvTranspose2d, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));
    input.client.execute(
        Box::new(kernel),
        &[
            &input.handle,
            &weight.handle,
            &bias_handle,
            &output.handle,
            &info_handle,
        ],
    );

    output
}

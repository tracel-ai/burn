use crate::{
    compute::StaticKernel,
    element::JitElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Element, ElementConversion, Shape,
};

kernel_wgsl!(Conv2d, "../../template/conv/conv2d.wgsl");

pub(crate) fn conv2d<R: Runtime, E: JitElement + Element>(
    input: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let input = kernel::into_contiguous(input);
    let weight = kernel::into_contiguous(weight);
    let [batch_size, _, in_height, in_width] = input.shape.dims;
    let [out_channels, _, kernel_0, kernel_1] = weight.shape.dims;

    let out_0 = calculate_conv_output_size(
        kernel_0,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_height,
    );
    let out_1 = calculate_conv_output_size(
        kernel_1,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_width,
    );

    let shape_out = Shape::new([batch_size, out_channels, out_0, out_1]);

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
        KernelSettings<Conv2d, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        output.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));

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

use crate::{
    element::WgpuElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};

kernel_wgsl!(Conv2d, "../../template/conv/conv2d.wgsl");

pub(crate) fn conv2d<E: WgpuElement>(
    input: WgpuTensor<E, 4>,
    weight: WgpuTensor<E, 4>,
    bias: Option<WgpuTensor<E, 1>>,
    options: ConvOptions<2>,
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

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
    let num_elems = shape_out.num_elements();

    let buffer = input
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), shape_out, buffer);

    let mut info = build_info(&[&input, &output, &weight]);
    info.push(options.stride[0] as u32);
    info.push(options.stride[1] as u32);
    info.push(options.padding[0] as u32);
    info.push(options.padding[1] as u32);
    info.push(options.dilation[0] as u32);
    info.push(options.dilation[1] as u32);
    info.push(options.groups as u32);

    let bias_buffer = bias
        .map(|bias| bias.buffer)
        .unwrap_or_else(|| input.context.create_buffer(core::mem::size_of::<E>()));

    let info_buffer = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = input
        .context
        .compile_static::<KernelSettings<Conv2d, E, i32, WORKGROUP, WORKGROUP, 1>>();

    input.context.execute(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        kernel,
        &[
            &input.buffer,
            &weight.buffer,
            &bias_buffer,
            &output.buffer,
            &info_buffer,
        ],
    );

    output
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{module, Distribution, Tensor};

    #[test]
    fn conv2d_should_work_with_multiple_invocations() {
        let input = Tensor::<TestBackend, 4>::random([6, 16, 32, 32], Distribution::Default);
        let weight = Tensor::<TestBackend, 4>::random([12, 8, 3, 3], Distribution::Default);
        let bias = Tensor::<TestBackend, 1>::random([12], Distribution::Default);
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());
        let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data());
        let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data());
        let options = burn_tensor::ops::ConvOptions::new([2, 3], [2, 3], [2, 3], 2);

        let output = module::conv2d(input, weight, Some(bias), options.clone());
        let output_ref = module::conv2d(input_ref, weight_ref, Some(bias_ref), options);

        output
            .into_data()
            .assert_approx_eq(&output_ref.into_data(), 3);
    }
}

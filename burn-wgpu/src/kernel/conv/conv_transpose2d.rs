use crate::{
    element::WgpuElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};
use burn_tensor::{ops::ConvTransposeOptions, Shape};

kernel_wgsl!(ConvTranspose2d, "../../template/conv/conv_transpose2d.wgsl");

pub(crate) fn conv_transpose2d<E: WgpuElement>(
    input: WgpuTensor<E, 4>,
    weight: WgpuTensor<E, 4>,
    bias: Option<WgpuTensor<E, 1>>,
    options: ConvTransposeOptions<2>,
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

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
        .compile_static::<KernelSettings<ConvTranspose2d, E, i32, WORKGROUP, WORKGROUP, 1>>();

    let workgroup = elemwise_workgroup(num_elems, WORKGROUP);
    input.context.execute(
        workgroup,
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
    use burn_tensor::{backend::Backend, module, Distribution, Tensor};

    #[test]
    fn conv_transpose2d_should_work_with_multiple_invocations() {
        TestBackend::seed(0);

        let height = 8;
        let width = 8;
        let in_channels = 8;
        let out_channels = 8;
        let batch_size = 32;
        let kernel_size_0 = 3;
        let kernel_size_1 = 3;
        let options =
            burn_tensor::ops::ConvTransposeOptions::new([1, 1], [1, 1], [0, 0], [1, 1], 1);

        let input = Tensor::<TestBackend, 4>::random(
            [batch_size, in_channels, height, width],
            Distribution::Default,
        );
        let weight = Tensor::<TestBackend, 4>::random(
            [
                in_channels,
                out_channels / options.groups,
                kernel_size_0,
                kernel_size_1,
            ],
            Distribution::Default,
        );
        let bias = Tensor::<TestBackend, 1>::random([out_channels], Distribution::Default);
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());
        let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data());
        let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data());

        let output = module::conv_transpose2d(input, weight, Some(bias), options.clone());
        let output_ref = module::conv_transpose2d(input_ref, weight_ref, Some(bias_ref), options);

        output
            .into_data()
            .assert_approx_eq(&output_ref.into_data(), 3);
    }
}

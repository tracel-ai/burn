use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};
use burn_tensor::{ops::ConvTransposeOptions, Element, ElementConversion, Shape};

kernel_wgsl!(ConvTranspose2d, "../../template/conv/conv_transpose2d.wgsl");

pub(crate) fn conv_transpose2d<E: WgpuElement + Element>(
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

    let kernel =
        StaticKernel::<KernelSettings<ConvTranspose2d, E, i32, WORKGROUP, WORKGROUP, 1>>::new(
            elemwise_workgroup(num_elems, WORKGROUP),
        );
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

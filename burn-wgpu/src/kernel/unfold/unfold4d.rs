use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};
use burn_tensor::{ops::UnfoldOptions, Element, Shape};

kernel_wgsl!(Unfold4d, "../../template/unfold/unfold4d.wgsl");

pub(crate) fn unfold4d<E: WgpuElement + Element>(
    input: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    options: UnfoldOptions,
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let intermediate_input = kernel::into_contiguous(input.clone());
    let [_, channels_in, _, _] = intermediate_input.shape.dims;
    let stride = options.stride.unwrap_or([1, 1]);
    let padding = options.padding.unwrap_or([0, 0]);
    let dilation = options.dilation.unwrap_or([1, 1]);

    let channels_out = channels_in * kernel_size[0] * kernel_size[1];

    let weight_shape = Shape::new([channels_out, channels_in, kernel_size[0], kernel_size[1]]);
    let weight = empty_device(
        intermediate_input.client.clone(),
        intermediate_input.device.clone(),
        weight_shape.clone(),
    );

    let mut info = build_info(&[&intermediate_input]);
    info.push(kernel_size[0] as u32);
    info.push(kernel_size[1] as u32);

    let info_handle = intermediate_input
        .client
        .create(bytemuck::cast_slice(&info));

    let kernel = StaticKernel::<KernelSettings<Unfold4d, E, i32, WORKGROUP, WORKGROUP, 1>>::new(
        elemwise_workgroup(weight.shape.num_elements(), WORKGROUP),
    );

    intermediate_input
        .client
        .execute(Box::new(kernel), &[&weight.handle, &info_handle]);

    let options = burn_tensor::ops::ConvOptions::new(stride, padding, dilation, 1);
    let unfolded = kernel::conv::conv2d(input, weight, None, options.clone());

    unfolded
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{module, Distribution, Tensor};

    #[test]
    fn unfold_should_match_reference_shape() {
        let input = Tensor::<TestBackend, 4>::random([6, 16, 32, 32], Distribution::Default);
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());
        let options = burn_tensor::ops::UnfoldOptions::new(Some(stride), None, None);

        let output = module::unfold4d(input, kernel_size, options.clone());
        let output_ref = module::unfold4d(input_ref, kernel_size, options);

        assert_eq!(output_ref.shape().dims, output.shape().dims);
    }
}

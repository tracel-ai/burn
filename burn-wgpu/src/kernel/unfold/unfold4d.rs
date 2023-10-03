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
) -> WgpuTensor<E, 3> {
    const WORKGROUP: usize = 32;

    let input = kernel::into_contiguous(input);
    let [batch_size, channels_in, in_height, in_width] = input.shape.dims;
    let stride = options.stride.unwrap_or([1, 1]);
    let padding = options.padding.unwrap_or([0, 0]);
    let dilation = options.dilation.unwrap_or([1, 1]);

    let channels_out = channels_in * kernel_size[0] * kernel_size[1];

    let l_dim_1 =
        (in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    let l_dim_2 =
        (in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
    let l = l_dim_1 * l_dim_2;

    let shape_out = Shape::new([batch_size, channels_out, l]);
    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
    );

    let mut info = build_info(&[&input]);
    info.push(kernel_size[0] as u32);
    info.push(kernel_size[1] as u32);
    info.push(stride[0] as u32);
    info.push(stride[1] as u32);
    info.push(padding[0] as u32);
    info.push(padding[1] as u32);
    info.push(dilation[0] as u32);
    info.push(dilation[1] as u32);

    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    let kernel = StaticKernel::<KernelSettings<Unfold4d, E, i32, WORKGROUP, WORKGROUP, 1>>::new(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
    );

    input.client.execute(
        Box::new(kernel),
        &[&input.handle, &output.handle, &info_handle],
    );

    output
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
        let options = burn_tensor::ops::UnfoldOptions::new(kernel_size, Some(stride), None, None);

        let output = module::unfold4d(input, kernel_size, options.clone());
        let output_ref = module::unfold4d(input_ref, kernel_size, options);

        assert_eq!(output_ref.shape().dims, output.shape().dims);
    }
}

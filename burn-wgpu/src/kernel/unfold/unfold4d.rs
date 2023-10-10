use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{self, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};
use burn_tensor::{ops::UnfoldOptions, Element, Shape};

kernel_wgsl!(FillWeights, "../../template/unfold/fill_weights.wgsl");

fn compute_unfolding_indices(channels_in: usize, kernel_size: [usize; 2]) -> Vec<usize> {
    let mut indices = vec![];

    for k in 0..channels_in {
        for i in 0..kernel_size[0] {
            for j in 0..kernel_size[1] {
                let output_channel = k * kernel_size[0] * kernel_size[1] + i * kernel_size[1] + j;
                let index = output_channel * channels_in * kernel_size[0] * kernel_size[1]
                    + k * kernel_size[0] * kernel_size[1]
                    + i * kernel_size[1]
                    + j;
                indices.push(index);
            }
        }
    }

    indices
}

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

    let weight_shape = Shape::new([
        channels_in * kernel_size[0] * kernel_size[1],
        channels_in,
        kernel_size[0],
        kernel_size[1],
    ]);
    let weight = kernel::into_contiguous(empty_device(
        intermediate_input.client.clone(),
        intermediate_input.device.clone(),
        weight_shape.clone(),
    ));

    let indices = compute_unfolding_indices(channels_in, kernel_size);
    let indices_handle = intermediate_input
        .client
        .create(bytemuck::cast_slice(&indices));

    let kernel = StaticKernel::<KernelSettings<FillWeights, E, i32, WORKGROUP, WORKGROUP, 1>>::new(
        elemwise_workgroup(weight.shape.num_elements(), WORKGROUP),
    );

    weight
        .client
        .execute(Box::new(kernel), &[&weight.handle, &indices_handle]);

    let options = burn_tensor::ops::ConvOptions::new(stride, padding, dilation, 1);
    kernel::conv::conv2d(input, weight, None, options.clone())
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{module, Distribution, Tensor};

    #[test]
    fn unfold_should_match_reference_values() {
        let input = Tensor::<TestBackend, 4>::random([6, 16, 32, 32], Distribution::Default);
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());
        let options = burn_tensor::ops::UnfoldOptions::new(Some(stride), None, None);

        let output = module::unfold4d(input, kernel_size, options.clone());
        let output_ref = module::unfold4d(input_ref, kernel_size, options);

        output
            .into_data()
            .assert_approx_eq(&output_ref.into_data(), 3);
    }
}

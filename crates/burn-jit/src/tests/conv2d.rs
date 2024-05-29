#[burn_tensor_testgen::testgen(conv2d)]
mod tests {
    use super::*;
    use burn_tensor::{module, Distribution, Tensor};

    #[test]
    fn conv2d_should_work_with_multiple_invocations() {
        let test_device = Default::default();
        let input =
            Tensor::<TestBackend, 4>::random([6, 16, 32, 32], Distribution::Default, &test_device);
        let weight =
            Tensor::<TestBackend, 4>::random([12, 8, 3, 3], Distribution::Default, &test_device);
        let bias = Tensor::<TestBackend, 1>::random([12], Distribution::Default, &test_device);
        let ref_device = Default::default();

        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &ref_device);
        let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);
        let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data(), &ref_device);

        let options = burn_tensor::ops::ConvOptions::new([2, 3], [2, 3], [2, 3], 2);

        let output = module::conv2d(input, weight, Some(bias), options.clone());
        let output_ref = module::conv2d(input_ref, weight_ref, Some(bias_ref), options);

        output
            .into_data()
            .assert_approx_eq(&output_ref.into_data(), 3);
    }

    #[test]
    fn conv2d_same_as_reference_configurable() {
        let test_device = Default::default();

        let height = 567;
        let width = 567;
        let batch_size = 1;
        let channels_in = 8;
        let groups = 1;
        let weight_channels_in = channels_in / groups;
        let channels_out = 6;

        let kernel_size_h = 3;
        let kernel_size_w = 3;
        let padding = 0;
        let stride = 1;
        let dilation = 1;

        let input = Tensor::<TestBackend, 4>::random(
            [batch_size, channels_in, height, width],
            Distribution::Default,
            &test_device,
        );
        let weight = Tensor::<TestBackend, 4>::random(
            [
                channels_out,
                weight_channels_in,
                kernel_size_h,
                kernel_size_w,
            ],
            Distribution::Default,
            &test_device,
        );

        let ref_device = Default::default();
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &ref_device);
        let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);

        let options = burn_tensor::ops::ConvOptions::new(
            [stride, stride],
            [padding, padding],
            [dilation, dilation],
            groups,
        );

        let output = module::conv2d(input, weight, None, options.clone());
        let output_ref = module::conv2d(input_ref, weight_ref, None, options);

        output
            .into_data()
            .assert_approx_eq(&output_ref.into_data(), 3);
    }
}

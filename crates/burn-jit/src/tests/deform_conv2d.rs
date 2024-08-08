#[burn_tensor_testgen::testgen(deform_conv2d)]
mod tests {
    use super::*;
    use burn_tensor::{module, Distribution, Tensor};

    #[test]
    fn deform_conv2d_should_work_with_multiple_invocations() {
        let test_device = Default::default();
        let input =
            Tensor::<TestBackend, 4>::random([6, 16, 32, 32], Distribution::Default, &test_device);
        let out_height = (32 + 2 * 2 - 2 * (2 - 1) - 1) / 2 + 1;
        let out_width = (32 + 2 * 3 - 3 * (3 - 1) - 1) / 3 + 1;
        let offset = Tensor::<TestBackend, 4>::random(
            [6, 2 * 3 * 2 * 3, out_height, out_width],
            Distribution::Default,
            &test_device,
        );
        let weight =
            Tensor::<TestBackend, 4>::random([12, 8, 3, 3], Distribution::Default, &test_device);
        let mask = Tensor::<TestBackend, 4>::random(
            [6, 3 * 2 * 3, out_height, out_width],
            Distribution::Default,
            &test_device,
        );
        let bias = Tensor::<TestBackend, 1>::random([12], Distribution::Default, &test_device);
        let ref_device = Default::default();

        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &ref_device);
        let offset_ref = Tensor::<ReferenceBackend, 4>::from_data(offset.to_data(), &ref_device);
        let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);
        let mask_ref = Tensor::<ReferenceBackend, 4>::from_data(mask.to_data(), &ref_device);
        let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data(), &ref_device);

        let options = burn_tensor::ops::DeformConvOptions::new([2, 3], [2, 3], [2, 3], 2, 3);

        let output = module::deform_conv2d(
            input,
            offset,
            weight,
            Some(mask),
            Some(bias),
            options.clone(),
        );
        let output_ref = module::deform_conv2d(
            input_ref,
            offset_ref,
            weight_ref,
            Some(mask_ref),
            Some(bias_ref),
            options,
        );

        output
            .into_data()
            .assert_approx_eq(&output_ref.into_data(), 3);
    }
}

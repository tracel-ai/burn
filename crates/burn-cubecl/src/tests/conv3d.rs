#[burn_tensor_testgen::testgen(conv3d)]
mod tests {
    use super::*;
    use burn_tensor::{module, Distribution, Tensor};

    #[test]
    fn conv3d_should_match_reference_backend() {
        let test_device = Default::default();
        let input = Tensor::<TestBackend, 5>::random(
            [6, 16, 32, 32, 32],
            Distribution::Default,
            &test_device,
        );
        let weight =
            Tensor::<TestBackend, 5>::random([12, 8, 3, 3, 3], Distribution::Default, &test_device);
        let bias = Tensor::<TestBackend, 1>::random([12], Distribution::Default, &test_device);
        let ref_device = Default::default();

        let input_ref = Tensor::<ReferenceBackend, 5>::from_data(input.to_data(), &ref_device);
        let weight_ref = Tensor::<ReferenceBackend, 5>::from_data(weight.to_data(), &ref_device);
        let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data(), &ref_device);

        let options = burn_tensor::ops::ConvOptions::new([2, 3, 4], [2, 3, 4], [2, 3, 4], 2);

        let output = module::conv3d(input, weight, Some(bias), options.clone());
        let output_ref = module::conv3d(input_ref, weight_ref, Some(bias_ref), options);

        output
            .into_data()
            .assert_approx_eq(&output_ref.into_data(), 3);
    }
}

#[burn_tensor_testgen::testgen(conv2d)]
mod tests {
    use super::*;
    use burn_cubecl::{
        kernel::{conv::nchw_to_nhwc, into_contiguous},
        tests::into_data_sync,
    };
    use burn_tensor::{backend::Backend, module, Distribution, Tensor};

    #[test]
    fn conv2d_should_match_reference_backend() {
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
    fn conv2d_should_match_reference_backend_implicit() {
        let test_device = Default::default();
        let input =
            Tensor::<TestBackend, 4>::random([4, 16, 6, 6], Distribution::Default, &test_device);
        let weight =
            Tensor::<TestBackend, 4>::random([16, 16, 3, 3], Distribution::Default, &test_device);
        let bias = Tensor::<TestBackend, 1>::random([16], Distribution::Default, &test_device);
        let ref_device = Default::default();

        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &ref_device);
        let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);
        let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data(), &ref_device);

        let options = burn_tensor::ops::ConvOptions::new([1, 1], [2, 2], [1, 1], 1);

        let output = module::conv2d(input, weight, Some(bias), options.clone());
        let output_ref = module::conv2d(input_ref, weight_ref, Some(bias_ref), options);

        output
            .into_data()
            .assert_approx_eq(&output_ref.into_data(), 2);
    }

    /// Regression test for bias loader in new implicit GEMM
    #[test]
    fn conv2d_should_match_reference_backend_bias_regression() {
        let test_device = Default::default();
        let input =
            Tensor::<TestBackend, 4>::random([1, 1, 1, 1], Distribution::Default, &test_device);
        let weight =
            Tensor::<TestBackend, 4>::random([32, 1, 3, 3], Distribution::Default, &test_device);
        let bias = Tensor::<TestBackend, 1>::random([32], Distribution::Default, &test_device);
        let ref_device = Default::default();

        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &ref_device);
        let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);
        let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data(), &ref_device);

        let options = burn_tensor::ops::ConvOptions::new([1, 1], [1, 1], [1, 1], 1);

        let output =
            module::conv2d(input, weight, Some(bias), options.clone()).permute([0, 2, 3, 1]);
        let output_ref =
            module::conv2d(input_ref, weight_ref, Some(bias_ref), options).permute([0, 2, 3, 1]);

        output
            .into_data()
            .assert_approx_eq(&output_ref.into_data(), 2);
    }

    #[test]
    fn nchw_to_nhwc_should_match_into_contiguous() {
        let test_device = Default::default();
        let input =
            Tensor::<TestBackend, 4>::random([4, 72, 53, 56], Distribution::Default, &test_device);

        type Float = <TestBackend as Backend>::FloatElem;

        let output = nchw_to_nhwc::<TestRuntime, Float>(input.clone().into_primitive().tensor());
        let output_ref = into_contiguous(
            input
                .clone()
                .permute([0, 2, 3, 1])
                .into_primitive()
                .tensor(),
        );

        into_data_sync::<TestRuntime, Float>(output)
            .assert_approx_eq(&into_data_sync::<TestRuntime, Float>(output_ref), 4);
    }

    /// Regression test for transpose kernel that was causing corruption with 17-64 in channels and
    /// at least 17 hw
    #[test]
    fn nchw_to_nhwc_should_match_into_contiguous_regression() {
        let test_device = Default::default();
        let input =
            Tensor::<TestBackend, 4>::random([1, 18, 17, 1], Distribution::Default, &test_device);

        type Float = <TestBackend as Backend>::FloatElem;

        let output = nchw_to_nhwc::<TestRuntime, Float>(input.clone().into_primitive().tensor());
        let output_ref = into_contiguous(
            input
                .clone()
                .permute([0, 2, 3, 1])
                .into_primitive()
                .tensor(),
        );

        into_data_sync::<TestRuntime, Float>(output)
            .assert_approx_eq(&into_data_sync::<TestRuntime, Float>(output_ref), 4);
    }
}

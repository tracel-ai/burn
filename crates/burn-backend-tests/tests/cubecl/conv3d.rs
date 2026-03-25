use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, module};

#[test]
fn conv3d_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<5>::random([6, 16, 32, 32, 32], Distribution::Default, &device);
    let weight = TestTensor::<5>::random([12, 8, 3, 3, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([12], Distribution::Default, &device);

    let input_ref = TestTensor::<5>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<5>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = burn_tensor::ops::ConvOptions::new([2, 3, 4], [2, 3, 4], [2, 3, 4], 2);

    let output = module::conv3d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv3d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

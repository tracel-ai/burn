use super::*;
use burn_tensor::{Distribution, Tolerance, module, ops::ConvOptions};

#[test]
fn conv1d_end_padding_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<3>::random([2, 4, 9], Distribution::Default, &device);
    let weight = TestTensor::<3>::random([6, 2, 4], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([6], Distribution::Default, &device);

    let input_ref = TestTensor::<3>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<3>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = ConvOptions::new_with_padding([2], [(0, 3)], [2], 2);
    let output = module::conv1d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv1d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

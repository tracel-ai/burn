use super::*;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;

#[test]
fn clamp_should_match_reference() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<4>::random([1, 5, 32, 32], Distribution::Default, &device);
    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);

    let output = input.clamp(0.3, 0.7);

    output.into_data().assert_approx_eq::<FloatElem>(
        &input_ref.clamp(0.3, 0.7).into_data(),
        Tolerance::default(),
    );
}

use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, Tensor};

#[test]
fn clamp_should_match_reference() {
    let input = Tensor::<TestBackend, 4>::random(
        [1, 5, 32, 32],
        Distribution::Default,
        &Default::default(),
    );
    let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &Default::default());

    let output = input.clamp(0.3, 0.7);

    output.into_data().assert_approx_eq::<FloatElem>(
        &input_ref.clamp(0.3, 0.7).into_data(),
        Tolerance::default(),
    );
}

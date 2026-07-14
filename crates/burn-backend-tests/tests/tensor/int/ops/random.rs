use super::*;
use burn_tensor::{Distribution, ElementConversion};

#[test]
fn rand_uniform_int() {
    let low = 0.;
    let high = 5.;

    let tensor = TestTensorInt::<1>::random(
        [100_000],
        Distribution::Uniform(low, high),
        &Default::default(),
    );

    tensor
        .into_data()
        .assert_within_range::<IntElem>(low.elem()..high.elem());
}

#[burn_tensor_testgen::testgen(inf)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, cast::ToElement};

    #[test]
    fn is_inf() {
        let no_inf = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let no_inf_expected =
            TestTensorBool::<2>::from([[false, false, false], [false, false, false]]);

        let with_inf =
            TestTensor::<2>::from([[0.0, f32::INFINITY, 2.0], [f32::NEG_INFINITY, 4.0, 5.0]]);
        let with_inf_expected =
            TestTensorBool::<2>::from([[false, true, false], [true, false, false]]);

        no_inf
            .is_inf()
            .into_data()
            .assert_eq(&no_inf_expected.into_data(), false);

        with_inf
            .is_inf()
            .into_data()
            .assert_eq(&with_inf_expected.into_data(), false);
    }
}

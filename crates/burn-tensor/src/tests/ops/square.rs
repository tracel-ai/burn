#[burn_tensor_testgen::testgen(square)]
mod tests {
    use super::*;
    use burn_tensor::{ElementConversion, Tensor, TensorData};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_support_sqrt_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.square();
        let expected = TensorData::from([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);

        output.into_data().assert_approx_eq::<FT>(
            &expected,
            Tolerance::relative(1e-4).set_half_precision_relative(1e-3),
        );
    }
}

#[burn_tensor_testgen::testgen(log_sigmoid)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, activation, cast::ToElement, tests::Numeric};

    #[test]
    fn test_log_sigmoid() {
        let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

        let output = activation::log_sigmoid(tensor);
        let expected = TensorData::from([[-3.132617e-1, -9.114665e-4], [-2.260327e-6, -3.0485873]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_log_sigmoid_numerical_stability() {
        let tensor = TestTensor::<1>::from([300.0, -300.0]);

        let output = activation::log_sigmoid(tensor);

        // For large negative values, the previous implementation −log(1 + exp(−x)) would give -inf
        let expected = TensorData::from([0.0, -300.0]);
        output.into_data().assert_approx_eq(&expected, 4);

        let tensor = TestTensor::<1>::from([
            <FloatType as Numeric>::max_value(),
            <FloatType as Numeric>::min_value(),
        ]);
        let output = activation::log_sigmoid(tensor);
        let expected = TensorData::from([0.0, <FloatType as Numeric>::min_value().to_f32()]);

        output.into_data().assert_approx_eq(&expected, 4);
    }
}

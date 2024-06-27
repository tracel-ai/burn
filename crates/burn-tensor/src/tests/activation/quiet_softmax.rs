#[burn_tensor_testgen::testgen(quiet_softmax)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_quiet_softmax_d2() {
        let tensor = TestTensor::from([[1.0, 7.0], [13.0, -3.0]]);

        let output = activation::quiet_softmax(tensor, 1);
        let expected = TensorData::from([[2.47e-03, 9.975e-01], [1.0, 1.1254e-07]]);

        output.into_data().assert_approx_eq(&expected, 4);
    }
}

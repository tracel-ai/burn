#[burn_tensor_testgen::testgen(hard_sigmoid)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_hard_sigmoid() {
        let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

        let output = activation::hard_sigmoid(tensor, 0.2, 0.5);
        let expected = TensorData::from([[0.7, 1.0], [1.0, 0.0]]);

        output.into_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    fn test_hard_sigmoid_overflow() {
        let tensor = TestTensor::<1>::from([f32::MAX, f32::MIN]);

        let output = activation::hard_sigmoid(tensor, 0.2, 0.5);
        let expected = TensorData::from([1.0, 0.0]);

        output.into_data().assert_approx_eq(&expected, 4);
    }
}

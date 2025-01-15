#[burn_tensor_testgen::testgen(cumsum)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Int, Tensor, TensorData};

    #[test]
    fn should_support_cumsum_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data(data, &device);

        let output = tensor.clone().cumsum(0);
        let expected = TensorData::from([[0.0, 1.0, 2.0], [3.0, 5.0, 7.0]]);

        output.into_data().assert_eq(&expected, false);

        let output = tensor.cumsum(1);
        let expected = TensorData::from([[0.0, 1.0, 3.0], [3.0, 7.0, 12.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_cumsum_ops_int() {
        let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
        let device = Default::default();
        let tensor = TestTensorInt::<2>::from_data(data, &device);

        let output = tensor.clone().cumsum(0);
        let expected = TensorData::from([[0, 1, 2], [3, 5, 7]]);

        output.into_data().assert_eq(&expected, false);

        let output = tensor.cumsum(1);
        let expected = TensorData::from([[0, 1, 3], [3, 7, 12]]);

        output.into_data().assert_eq(&expected, false);
    }
}

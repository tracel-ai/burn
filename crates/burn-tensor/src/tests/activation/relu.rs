#[burn_tensor_testgen::testgen(relu)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_relu_d2() {
        let tensor = TestTensor::from([[0.0, -1.0, 2.0], [3.0, -4.0, 5.0]]);

        let data_actual = activation::relu(tensor).into_data();

        let data_expected = TensorData::from([[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]]);
        assert_eq!(data_expected, data_actual);
    }
}

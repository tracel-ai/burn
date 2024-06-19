#[burn_tensor_testgen::testgen(relu)]
mod tests {
    use super::*;
    use burn_tensor::{activation, backend::Backend, Tensor, TensorData};

    #[test]
    fn test_relu_d2() {
        let tensor = TestTensor::from([[0.0, -1.0, 2.0], [3.0, -4.0, 5.0]]);

        let output = activation::relu(tensor);
        let expected = TensorData::from([[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]])
            .convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_eq(&expected, true);
    }
}

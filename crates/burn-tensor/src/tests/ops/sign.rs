#[burn_tensor_testgen::testgen(sign)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Tensor, TensorData};

    #[test]
    fn should_support_sign_ops_float() {
        let tensor = TestTensor::from([[-0.2, -1.0, 2.0], [3.0, 0.0, -5.0]]);

        let output = tensor.sign();
        let expected = TensorData::from([[-1.0, -1.0, 1.0], [1.0, 0.0, -1.0]])
            .convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn should_support_sign_ops_int() {
        let tensor = TestTensorInt::from([[-2, -1, 2], [3, 0, -5]]);

        let output = tensor.sign();
        let expected = TensorData::from([[-1, -1, 1], [1, 0, -1]])
            .convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }
}

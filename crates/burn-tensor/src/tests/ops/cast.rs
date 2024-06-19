#[burn_tensor_testgen::testgen(cast)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Bool, Tensor, TensorData};

    #[test]
    fn cast_float_to_int() {
        let tensor = TestTensor::from([[1.0, 2.0, 3.0], [4.4, 5.5, 6.6]]).int();
        let expected =
            TensorData::from([[1, 2, 3], [4, 5, 6]]).convert::<<TestBackend as Backend>::IntElem>();

        tensor.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn cast_int_to_float_tensor() {
        let tensor = TestTensorInt::from([[1, 2, 3], [4, 5, 6]]).float();

        let expected = TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            .convert::<<TestBackend as Backend>::FloatElem>();

        tensor.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn cast_bool_to_int_tensor() {
        let tensor = TestTensorBool::from([[true, false, true], [false, false, true]]).int();

        let expected =
            TensorData::from([[1, 0, 1], [0, 0, 1]]).convert::<<TestBackend as Backend>::IntElem>();

        tensor.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn cast_bool_to_float_tensor() {
        let tensor =
            Tensor::<TestBackend, 2, Bool>::from([[true, false, true], [false, false, true]])
                .float();

        let expected = TensorData::from([[1., 0., 1.], [0., 0., 1.]])
            .convert::<<TestBackend as Backend>::FloatElem>();

        tensor.into_data().assert_eq(&expected, true);
    }
}

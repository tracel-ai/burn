#[burn_tensor_testgen::testgen(arg)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Tensor, TensorData};

    #[test]
    fn test_argmax_2d_dim0() {
        let tensor = TestTensor::from([[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.argmax(0);
        let expected = TensorData::from([[0, 0, 1]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_argmin_2d_dim0() {
        let tensor = TestTensor::from([[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]);

        let output = tensor.argmin(0);
        let expected = TensorData::from([[0, 1, 0]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_argmax_2d_dim0_int() {
        let tensor = TestTensorInt::from([[10, 11, 2], [3, 4, 5]]);

        let output = tensor.argmax(0);
        let expected = TensorData::from([[0, 0, 1]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_argmin_2d_dim0_int() {
        let tensor = TestTensorInt::from([[10, 11, 2], [30, 4, 5]]);

        let output = tensor.argmin(0);
        let expected = TensorData::from([[0, 1, 0]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_argmax_2d_dim1() {
        let tensor = TestTensor::from([[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.argmax(1);
        let expected = TensorData::from([[1], [2]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_argmin_2d_dim1() {
        let tensor = TestTensor::from([[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]);

        let output = tensor.argmin(1);
        let expected = TensorData::from([[2], [1]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }
}

#[burn_tensor_testgen::testgen(matmul)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_matmul_d2() {
        let tensor_1 = TestTensor::from_floats([[1.0, 7.0], [2.0, 3.0], [1.0, 5.0]]);
        let tensor_2 = TestTensor::from_floats([[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]]);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[18.0, 28.0, 40.0], [14.0, 23.0, 25.0], [14.0, 22.0, 30.0]])
        );
    }

    #[test]
    fn test_matmul_d3() {
        let tensor_1 = TestTensor::from_floats([[[1.0, 7.0], [2.0, 3.0]]]);
        let tensor_2 = TestTensor::from_floats([[[4.0, 7.0], [2.0, 3.0]]]);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[[18.0, 28.0], [14.0, 23.0]]])
        );
    }

    #[test]
    fn test_matmul_broadcast_1() {
        let tensor_1 = TestTensor::from_floats([[[1.0, 7.0], [2.0, 3.0]]]);
        let tensor_2 =
            TestTensor::from_floats([[[4.0, 7.0], [2.0, 3.0]], [[2.0, 5.0], [6.0, 3.0]]]);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[[18.0, 28.0], [14.0, 23.0]], [[44.0, 26.0], [22.0, 19.0]]])
        );
    }
}

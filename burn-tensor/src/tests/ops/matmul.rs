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

    #[test]
    fn test_matmul_simple_1() {
        let tensor_1 = TestTensor::from_floats([[5.0, 14.0], [14.0, 50.0]]);
        let tensor_2 = TestTensor::from_floats([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[15.0, 34.0, 53.0], [42.0, 106.0, 170.0]])
        );
    }

    #[test]
    fn test_matmul_simple_2() {
        let tensor_1 = TestTensor::from_floats([[1.0, 2.0, 3.0, 4.0]]);
        let tensor_2 = TestTensor::from_floats([[3.0], [4.0], [5.0], [6.0]]);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(tensor_3.into_data(), Data::from([[50.0]]));
    }

    #[test]
    fn test_matmul_simple_3() {
        let tensor_1 =
            TestTensor::from_floats([[3., 3., 3.], [4., 4., 4.], [5., 5., 5.], [6., 6., 6.]]);
        let tensor_2 =
            TestTensor::from_floats([[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]]);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([
                [9., 18., 27., 36.],
                [12., 24., 36., 48.],
                [15., 30., 45., 60.],
                [18., 36., 54., 72.]
            ])
        );
    }
}

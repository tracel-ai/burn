#[burn_tensor_testgen::testgen(cross)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, backend::Backend};

    #[test]
    fn test_cross_3d_last_dim() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0], [2.0, -1.0, 4.0]]);
        let tensor_2 = TestTensor::from([[4.0, -2.0, 1.0], [3.0, 5.0, -2.0]]);

        let output = tensor_1.cross(tensor_2, -1);

        output.into_data().assert_eq(
            &TensorData::from([[-7.0, -21.0, -14.0], [-18.0, 16.0, 13.0]]),
            false,
        );
    }

    #[test]
    fn test_cross_3d_dim0() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);
        let tensor_2 = TestTensor::from([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]);

        let output = tensor_1.cross(tensor_2, 0);

        output.into_data().assert_eq(
            &TensorData::from([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
            false,
        );
    }

    #[test]
    fn test_cross_3d_broadcast() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0]]);
        let tensor_2 = TestTensor::from([[4.0, -2.0, 1.0], [3.0, 5.0, -2.0]]);

        let output = tensor_1.cross(tensor_2, -1);

        output.into_data().assert_eq(
            &TensorData::from([[-7.0, -21.0, -14.0], [19.0, -13.0, -4.0]]),
            false,
        );
    }

    #[test]
    fn test_cross_4d_last_dim() {
        let tensor_1 = TestTensor::<3>::from([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        let tensor_2 = TestTensor::from([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]);

        let output = tensor_1.cross(tensor_2, -1);

        output.into_data().assert_eq(
            &TensorData::from([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]),
            false,
        );
    }
}

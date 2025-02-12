#[burn_tensor_testgen::testgen(select)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_select_1d() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data([0.0, 1.0, 2.0], &device);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

        let output = tensor.select(0, indices);
        let expected = TensorData::from([1.0, 1.0, 0.0, 1.0, 2.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_1d_int() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::from_data([5, 6, 7], &device);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

        let output = tensor.select(0, indices);
        let expected = TensorData::from([6, 6, 5, 6, 7]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim0_same_num_dim() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
        let indices = TestTensorInt::from_data(([1, 0]), &device);

        let output = tensor.select(0, indices);
        let expected = TensorData::from([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim0_more_num_dim() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
        let indices = TestTensorInt::from_data([1, 0, 1, 1], &device);

        let output = tensor.select(0, indices);
        let expected = TensorData::from([
            [3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim0_vec() {
        let device = Default::default();
        let tensor =
            TestTensor::<2>::from_data([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], &device);
        let indices = TestTensorInt::from_data([1, 0, 3, 2], &device);

        let output = tensor.select(0, indices);
        let expected = TensorData::from([[2.0, 3.0], [0.0, 1.0], [6.0, 7.0], [4.0, 5.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim1() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

        let output = tensor.select(1, indices);
        let expected = TensorData::from([[1.0, 1.0, 0.0, 1.0, 2.0], [4.0, 4.0, 3.0, 4.0, 5.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_assign_1d() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data([0.0, 1.0, 2.0], &device);
        let values = TestTensor::from_data([5.0, 4.0, 3.0, 2.0, 1.0], &device);
        let indices = TestTensorInt::from_data(TensorData::from([1, 1, 0, 1, 2]), &device);

        let output = tensor.select_assign(0, indices, values);
        let expected = TensorData::from([3.0, 12.0, 3.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_assign_1d_int() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::from_data([7, 8, 9], &device);
        let values = TestTensorInt::from_data([5, 4, 3, 2, 1], &device);
        let indices = TestTensorInt::from_data(TensorData::from([1, 1, 0, 1, 2]), &device);

        let output = tensor.select_assign(0, indices, values);
        let expected = TensorData::from([10, 19, 10]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_assign_2d_dim0() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
        let values = TestTensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let indices = TestTensorInt::from_data(TensorData::from([1, 0]), &device);

        let output = tensor.select_assign(0, indices, values);
        let expected = TensorData::from([[4.0, 6.0, 8.0], [4.0, 6.0, 8.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_assign_2d_dim1() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
        let values = TestTensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let indices = TestTensorInt::from_data(TensorData::from([1, 0, 2]), &device);

        let output = tensor.select_assign(1, indices, values);
        let expected = TensorData::from([[2.0, 2.0, 5.0], [8.0, 8.0, 11.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic]
    fn should_select_panic_invalid_dimension() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

        tensor.select(10, indices);
    }
}

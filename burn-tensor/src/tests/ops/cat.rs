#[burn_tensor_testgen::testgen(cat)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{Bool, Data, Int, Tensor};
    #[test]
    fn should_support_cat_ops_2d_dim0() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_data([[1.0, 2.0, 3.0]], &device);
        let tensor_2 = TestTensor::from_data([[4.0, 5.0, 6.0]], &device);

        let data_actual = TestTensor::cat(vec![tensor_1, tensor_2], 0).into_data();

        let data_expected = Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_cat_ops_int() {
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2, Int>::from_data([[1, 2, 3]], &device);
        let tensor_2 = Tensor::<TestBackend, 2, Int>::from_data([[4, 5, 6]], &device);

        let data_actual = Tensor::cat(vec![tensor_1, tensor_2], 0).into_data();

        let data_expected = Data::from([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(&data_actual, &data_expected);
    }

    #[test]
    fn should_support_cat_ops_bool() {
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2, Bool>::from_data([[false, true, true]], &device);
        let tensor_2 = Tensor::<TestBackend, 2, Bool>::from_data([[true, true, false]], &device);

        let data_actual = Tensor::cat(vec![tensor_1, tensor_2], 0).into_data();

        let data_expected = Data::from([[false, true, true], [true, true, false]]);
        assert_eq!(&data_actual, &data_expected);
    }

    #[test]
    fn should_support_cat_ops_2d_dim1() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_data([[1.0, 2.0, 3.0]], &device);
        let tensor_2 = TestTensor::from_data([[4.0, 5.0, 6.0]], &device);

        let data_actual = TestTensor::cat(vec![tensor_1, tensor_2], 1).into_data();

        let data_expected = Data::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_cat_ops_3d() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], &device);
        let tensor_2 = TestTensor::from_data([[[4.0, 5.0, 6.0]]], &device);

        let data_actual = TestTensor::cat(vec![tensor_1, tensor_2], 0).into_data();

        let data_expected = Data::from([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]], [[4.0, 5.0, 6.0]]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_dimensions_are_not_the_same() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_data([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], &device);
        let tensor_2 = TestTensor::from_data([[4.0, 5.0]], &device);

        TestTensor::cat(vec![tensor_1, tensor_2], 0).into_data();
    }

    #[test]
    #[should_panic]
    fn should_panic_when_list_of_vectors_is_empty() {
        let tensor: Vec<TestTensor<2>> = vec![];
        TestTensor::cat(tensor, 0).into_data();
    }

    #[test]
    #[should_panic]
    fn should_panic_when_cat_exceeds_dimension() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]], &device);
        let tensor_2 = TestTensor::from_data([[[4.0, 5.0, 6.0]]], &device);

        TestTensor::cat(vec![tensor_1, tensor_2], 3).into_data();
    }
}

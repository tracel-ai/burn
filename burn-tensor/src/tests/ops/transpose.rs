#[burn_tensor_testgen::testgen(transpose)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Data, Int, Tensor};

    #[test]
    fn should_support_transpose_ops() {
        let tensor = TestTensor::from_floats_devauto([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]);

        let data_actual = tensor.transpose().into_data();

        let data_expected = Data::from([
            [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
            [[6.0, 9.0], [7.0, 10.0], [8.0, 11.0]],
        ]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_swap_dims() {
        let tensor = TestTensor::from_floats_devauto([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]);

        let data_actual = tensor.swap_dims(0, 2).into_data();

        let data_expected = Data::from([
            [[0.0, 6.0], [3.0, 9.0]],
            [[1.0, 7.0], [4.0, 10.0]],
            [[2.0, 8.0], [5.0, 11.0]],
        ]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_transpose_ops_int() {
        let tensor = Tensor::<TestBackend, 3, Int>::from_data_devauto([
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
        ]);

        let data_actual = tensor.transpose().into_data();

        let data_expected = Data::from([[[0, 3], [1, 4], [2, 5]], [[6, 9], [7, 10], [8, 11]]]);
        assert_eq!(&data_expected, &data_actual);
    }

    #[test]
    fn should_support_swap_dims_int() {
        let tensor = Tensor::<TestBackend, 3, Int>::from_data_devauto([
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
        ]);

        let data_actual = tensor.swap_dims(0, 2).into_data();

        let data_expected = Data::from([[[0, 6], [3, 9]], [[1, 7], [4, 10]], [[2, 8], [5, 11]]]);
        assert_eq!(&data_expected, &data_actual);
    }

    #[test]
    fn should_support_transpose_bool() {
        let tensor = Tensor::<TestBackend, 3, Bool>::from_data_devauto([
            [[false, true, false], [false, false, false]],
            [[false, false, true], [false, false, true]],
        ]);

        let data_actual = tensor.transpose().into_data();

        let data_expected = Data::from([
            [[false, false], [true, false], [false, false]],
            [[false, false], [false, false], [true, true]],
        ]);
        assert_eq!(&data_expected, &data_actual);
    }

    #[test]
    fn should_support_swap_dims_bool() {
        let tensor = Tensor::<TestBackend, 3, Bool>::from_data_devauto([
            [[false, true, false], [false, false, false]],
            [[false, false, true], [false, false, true]],
        ]);

        let data_actual = tensor.swap_dims(0, 2).into_data();

        let data_expected = Data::from([
            [[false, false], [false, false]],
            [[true, false], [false, false]],
            [[false, true], [false, true]],
        ]);
        assert_eq!(&data_expected, &data_actual);
    }
}

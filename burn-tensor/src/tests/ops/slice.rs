#[burn_tensor_testgen::testgen(slice)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_full_sliceing_1d() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone());

        let data_actual = tensor.slice([0..3]).into_data();

        assert_eq!(data, data_actual);
    }

    #[test]
    fn should_support_partial_sliceing_1d() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data);

        let data_actual = tensor.slice([1..3]).into_data();

        let data_expected = Data::from([1.0, 2.0]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_full_sliceing_2d() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data.clone());

        let data_actual_1 = tensor.clone().slice([0..2]).into_data();
        let data_actual_2 = tensor.slice([0..2, 0..3]).into_data();

        assert_eq!(data, data_actual_1);
        assert_eq!(data, data_actual_2);
    }

    #[test]
    fn should_support_partial_sliceing_2d() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.slice([0..2, 0..2]).into_data();

        let data_expected = Data::from([[0.0, 1.0], [3.0, 4.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_partial_sliceing_3d() {
        let tensor = TestTensor::from_floats([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]);

        let data_actual = tensor.slice([1..2, 1..2, 0..2]).into_data();

        let data_expected = Data::from([[[9.0, 10.0]]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_partial_sliceing_3d_non_contiguous() {
        let tensor = TestTensor::from_floats([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]);

        let data_actual = tensor.transpose().slice([1..2, 1..2, 0..2]).into_data();

        let data_expected = Data::from([[[7.0, 10.0]]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_slice_assign_1d() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let data_assigned = Data::from([10.0, 5.0]);

        let tensor = Tensor::<TestBackend, 1>::from_data(data);
        let tensor_assigned = Tensor::<TestBackend, 1>::from_data(data_assigned);

        let data_actual = tensor.slice_assign([0..2], tensor_assigned).into_data();

        let data_expected = Data::from([10.0, 5.0, 2.0]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_slice_assign_2d() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_assigned = Data::from([[10.0, 5.0]]);

        let tensor = Tensor::<TestBackend, 2>::from_data(data);
        let tensor_assigned = Tensor::<TestBackend, 2>::from_data(data_assigned);

        let data_actual = tensor
            .slice_assign([1..2, 0..2], tensor_assigned)
            .into_data();

        let data_expected = Data::from([[0.0, 1.0, 2.0], [10.0, 5.0, 5.0]]);
        assert_eq!(data_expected, data_actual);
    }
}

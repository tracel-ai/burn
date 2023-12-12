#[burn_tensor_testgen::testgen(slice)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

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

    #[test]
    fn slice_should_not_corrupt_potentially_inplace_operations() {
        let tensor = Tensor::<TestBackend, 1, Int>::from_data([1, 2, 3, 4, 5]);
        let tensor = tensor.clone().slice([0..3]) + tensor.clone().slice([2..5]);

        assert_eq!(tensor.into_data(), Data::from([4, 6, 8]));
    }

    #[test]
    fn slice_assign_should_not_corrupt_potentially_inplace_operations() {
        let tensor = Tensor::<TestBackend, 1, Int>::from_data([1, 2, 3, 4, 5]);
        let values = Tensor::<TestBackend, 1, Int>::from_data([10, 20, 30]);
        let tensor_1 = tensor.clone().slice_assign([0..3], values);
        let tensor_2 = tensor + 2;

        assert_eq!(tensor_1.into_data(), Data::from([10, 20, 30, 4, 5]));
        assert_eq!(tensor_2.into_data(), Data::from([3, 4, 5, 6, 7]));
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_exceeds_dimension() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone());

        let data_actual = tensor.slice([0..4]).into_data();

        assert_eq!(data, data_actual);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_with_too_many_dimensions() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone());

        let data_actual = tensor.slice([0..1, 0..1]).into_data();

        assert_eq!(data, data_actual);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_is_desc() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone());

        #[allow(clippy::reversed_empty_ranges)]
        let data_actual = tensor.slice([2..1]).into_data();

        assert_eq!(data, data_actual);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_is_equal() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone());

        let data_actual = tensor.slice([1..1]).into_data();

        assert_eq!(data, data_actual);
    }
}

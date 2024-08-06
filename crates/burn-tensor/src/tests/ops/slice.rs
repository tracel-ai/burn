#[burn_tensor_testgen::testgen(slice)]
mod tests {
    use super::*;
    use burn_tensor::{Int, Tensor, TensorData};

    #[test]
    fn should_support_full_sliceing_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([0..3]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    fn should_support_partial_sliceing_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &Default::default());

        let output = tensor.slice([1..3]);
        let expected = TensorData::from([1.0, 2.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_full_sliceing_2d() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data.clone(), &Default::default());

        let output = tensor.clone().slice([0..2]);
        output.into_data().assert_eq(&data, false);

        let output = tensor.slice([0..2, 0..3]);
        output.into_data().assert_eq(&data, false);
    }

    #[test]
    fn should_support_partial_sliceing_2d() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.slice([0..2, 0..2]);
        let expected = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_partial_sliceing_3d() {
        let tensor = TestTensor::<3>::from_floats(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ],
            &Default::default(),
        );

        let output = tensor.slice([1..2, 1..2, 0..2]);
        let expected = TensorData::from([[[9.0, 10.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_partial_sliceing_3d_non_contiguous() {
        let tensor = TestTensor::<3>::from_floats(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ],
            &Default::default(),
        );

        let output = tensor.transpose().slice([1..2, 1..2, 0..2]);
        let expected = TensorData::from([[[7.0, 10.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_assign_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let data_assigned = TensorData::from([10.0, 5.0]);

        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);
        let tensor_assigned = Tensor::<TestBackend, 1>::from_data(data_assigned, &device);

        let output = tensor.slice_assign([0..2], tensor_assigned);
        let expected = TensorData::from([10.0, 5.0, 2.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_assign_2d() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_assigned = TensorData::from([[10.0, 5.0]]);

        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &device);
        let tensor_assigned = Tensor::<TestBackend, 2>::from_data(data_assigned, &device);

        let output = tensor.slice_assign([1..2, 0..2], tensor_assigned);
        let expected = TensorData::from([[0.0, 1.0, 2.0], [10.0, 5.0, 5.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn slice_should_not_corrupt_potentially_inplace_operations() {
        let tensor = Tensor::<TestBackend, 1, Int>::from_data([1, 2, 3, 4, 5], &Default::default());
        let tensor = tensor.clone().slice([0..3]) + tensor.clone().slice([2..5]);

        let expected = TensorData::from([4, 6, 8]);

        tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn slice_assign_should_not_corrupt_potentially_inplace_operations() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::from_data([1, 2, 3, 4, 5], &device);
        let values = Tensor::<TestBackend, 1, Int>::from_data([10, 20, 30], &device);
        let tensor_1 = tensor.clone().slice_assign([0..3], values);
        let tensor_2 = tensor + 2;

        let expected = TensorData::from([10, 20, 30, 4, 5]);

        tensor_1.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([3, 4, 5, 6, 7]);

        tensor_2.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn clamp_when_slice_exceeds_dimension() {
        let data = TensorData::from([0.0f32, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([0..4]);
        output.into_data().assert_eq(&data, true);
    }

    #[test]
    fn negative_dimensions() {
        let data = TensorData::from([[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data.clone(), &Default::default());

        // Clamping to the tensor dimensions
        let output = tensor.clone().slice([(0, 4), (0, 4)]);
        output.into_data().assert_eq(&data, true);

        // Negative dimensions
        let output = tensor.clone().slice([(0, 1), (0, 1)]);
        let data = TensorData::from([[0.0f32]]);
        output.into_data().assert_eq(&data, true);

        let output = tensor.slice([(0, -1), (0, -2)]);
        output.into_data().assert_eq(&data, true);
    }

    #[test]
    fn missing_dimensions() {
        let data = TensorData::from([[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data.clone(), &Default::default());

        // Clamping to the tensor dimensions
        let output = tensor.clone().slice([Some((0, 4)), Some((0, 4))]);
        output.into_data().assert_eq(&data, true);

        // Negative dimensions
        let data = TensorData::from([[0.0f32]]);
        let output = tensor.clone().slice([Some((0, -1)), Some((0, -2))]);
        output.into_data().assert_eq(&data, true);

        // Missing dimensions
        let output = tensor.clone().slice([Some((0, 1)), None]);
        let data = TensorData::from([[0.0f32, 1.0, 2.0]]);
        output.into_data().assert_eq(&data, true);

        let output = tensor.clone().slice([None, Some((0, 2))]);
        let data = TensorData::from([[0.0f32, 1.0], [3.0, 4.0]]);
        output.into_data().assert_eq(&data, true);

        let output = tensor.clone().slice([None, None]);
        let data = TensorData::from([[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        output.into_data().assert_eq(&data, true);
    }

    #[test]
    fn should_slice_aggregation_result() {
        // Some backends (e.g., tch) tensor primitive results in 0-dim tensor for aggregation
        let tensor = TestTensor::<1>::from([0.0, 1.0, 2.0]).mean();

        let output = tensor.clone().slice([(0..1)]);
        output.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_with_too_many_dimensions() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([0..1, 0..1]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_is_desc() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone(), &Default::default());

        #[allow(clippy::reversed_empty_ranges)]
        let output = tensor.slice([2..1]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_is_equal() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([1..1]);

        output.into_data().assert_eq(&data, false);
    }
}

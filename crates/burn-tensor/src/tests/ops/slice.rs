#[burn_tensor_testgen::testgen(slice)]
mod tests {
    use super::*;
    use burn_tensor::{Int, Slice, Tensor, TensorData, as_type, s};

    #[test]
    fn should_support_slice_dim_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice_dim(0, -2..);
        output
            .into_data()
            .assert_eq(&TensorData::from([1.0, 2.0]), false);
    }

    #[test]
    #[should_panic(expected = "The provided dimension exceeds the tensor dimensions")]
    fn should_panic_when_slice_dim_1d_bad_dim() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        let _output = tensor.slice_dim(1, 1..);
    }

    #[test]
    fn should_support_slice_dim_2d() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        let output = tensor.slice_dim(1, 1..);
        output
            .into_data()
            .assert_eq(&TensorData::from([[1.0, 2.0], [4.0, 5.0]]), false);
    }

    #[test]
    fn should_support_full_sliceing_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([0..3]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    fn should_support_partial_sliceing_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.slice([1..3]);
        let expected = TensorData::from([1.0, 2.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_full_sliceing_2d() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        let output = tensor.clone().slice([0..2]);
        output.into_data().assert_eq(&data, false);

        let output = tensor.slice([0..2, 0..3]);
        output.into_data().assert_eq(&data, false);
    }

    #[test]
    fn should_support_partial_sliceing_2d() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.slice([0..2, 0..2]);
        let expected = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_range_first_dim() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.slice(0..1);
        let expected = TensorData::from([[0.0, 1.0, 2.0]]);

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
        let tensor = TestTensor::<1>::from_data(data, &device);
        let tensor_assigned = TestTensor::<1>::from_data(data_assigned, &device);

        let output = tensor.slice_assign([0..2], tensor_assigned);
        let expected = TensorData::from([10.0, 5.0, 2.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_assign_2d() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_assigned = TensorData::from([[10.0, 5.0]]);

        let device = Default::default();
        let tensor = TestTensor::<2>::from_data(data, &device);
        let tensor_assigned = TestTensor::<2>::from_data(data_assigned, &device);

        let output = tensor.slice_assign([1..2, 0..2], tensor_assigned);
        let expected = TensorData::from([[0.0, 1.0, 2.0], [10.0, 5.0, 5.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_fill_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);

        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.slice_fill([0..2], -1.0);
        let expected = TensorData::from([-1.0, -1.0, 2.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_fill_1d_neg() {
        let data = TensorData::from([0.0, 1.0, 2.0]);

        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.slice_fill([-1..], -1.0);
        let expected = TensorData::from([0.0, 1.0, -1.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_fill_2d() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let device = Default::default();
        let tensor = TestTensor::<2>::from_data(data, &device);

        let output = tensor.slice_fill([1..2, 0..2], -1.0);
        let expected = TensorData::from([[0.0, 1.0, 2.0], [-1.0, -1.0, 5.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic(expected = "slice_fill does not support steps != 1 yet")]
    fn slice_fill_should_panic_with_non_unit_step() {
        let device = Default::default();
        let tensor = TestTensor::<2>::ones([4, 4], &device);

        // This should panic because slice_fill doesn't support steps != 1
        let _result = tensor.slice_fill(s![0..4;2, ..], 5.0);
    }

    #[test]
    fn slice_should_not_corrupt_potentially_inplace_operations() {
        let tensor = TestTensorInt::<1>::from_data([1, 2, 3, 4, 5], &Default::default());
        let tensor = tensor.clone().slice([0..3]) + tensor.clone().slice([2..5]);

        let expected = TensorData::from([4, 6, 8]);

        tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn slice_assign_should_not_corrupt_potentially_inplace_operations() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::from_data([1, 2, 3, 4, 5], &device);
        let values = TestTensorInt::<1>::from_data([10, 20, 30], &device);
        let tensor_1 = tensor.clone().slice_assign([0..3], values);
        let tensor_2 = tensor + 2;

        let expected = TensorData::from([10, 20, 30, 4, 5]);

        tensor_1.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([3, 4, 5, 6, 7]);

        tensor_2.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic(expected = "slice_assign does not support steps != 1 yet")]
    fn slice_assign_should_panic_with_non_unit_step() {
        let device = Default::default();
        // Create tensors where the shapes would match if steps were supported
        let tensor = TestTensor::<2>::ones([4, 4], &device);
        // With step=2 on first dim, we'd select indices 0 and 2, so we need a [2, 4] values tensor
        // But since steps aren't supported, this should panic before shape validation
        let values = TestTensor::<2>::zeros([2, 4], &device);

        // This should panic because slice_assign doesn't support steps != 1
        // We use s! macro to create a slice with step=2
        let _result = tensor.slice_assign(s![0..3;2, ..], values);
    }

    #[test]
    fn clamp_when_slice_exceeds_dimension() {
        let data = TensorData::from(as_type!(FloatType: [0.0f32, 1.0, 2.0]));
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([0..4]);
        output.into_data().assert_eq(&data, true);
    }

    #[test]
    fn negative_dimensions() {
        let data = TensorData::from(as_type!(FloatType: [[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        // Clamping to the tensor dimensions
        let output = tensor.clone().slice([0..4, 0..4]);
        output.into_data().assert_eq(&data, true);

        // Negative dimensions
        let output = tensor.clone().slice([0..1, 0..1]);
        let data = TensorData::from(as_type!(FloatType: [[0.0f32]]));
        output.into_data().assert_eq(&data, true);

        let output = tensor.slice(s![0..-1, 0..-2]);
        output.into_data().assert_eq(&data, true);
    }

    #[test]
    fn missing_dimensions() {
        let data = TensorData::from(as_type!(FloatType: [[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        // Clamping to the tensor dimensions
        let output = tensor.clone().slice([0..4, 0..4]);
        output.into_data().assert_eq(&data, true);

        // Negative dimensions
        let data = TensorData::from(as_type!(FloatType: [[0.0f32]]));
        let output = tensor.clone().slice(s![0..-1, 0..-2]);
        output.into_data().assert_eq(&data, true);

        // Missing dimensions
        let output = tensor.clone().slice(s![0..1, ..]);
        let data = TensorData::from(as_type!(FloatType: [[0.0f32, 1.0, 2.0]]));
        output.into_data().assert_eq(&data, true);

        let output = tensor.clone().slice(s![.., 0..2]);
        let data = TensorData::from(as_type!(FloatType: [[0.0f32, 1.0], [3.0, 4.0]]));
        output.into_data().assert_eq(&data, true);

        let output = tensor.clone().slice([.., ..]);
        let data = TensorData::from(as_type!(FloatType: [[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]));
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
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([0..1, 0..1]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_is_desc() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice(s![2..1]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_is_equal() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([1..1]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    fn test_slice_with_positive_step() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        );

        // Test step=2 along first dimension
        let sliced = tensor.clone().slice([s![0..3;2]]);
        let expected = TensorData::from([[1.0, 2.0, 3.0, 4.0], [9.0, 10.0, 11.0, 12.0]]);
        sliced.into_data().assert_eq(&expected, false);

        // Test step=2 along second dimension
        let sliced = tensor.clone().slice(s![.., 0..4;2]);
        let expected = TensorData::from([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]);
        sliced.into_data().assert_eq(&expected, false);

        // Test step=2 along both dimensions
        let sliced = tensor.clone().slice(s![0..3;2, 0..4;2]);
        let expected = TensorData::from([[1.0, 3.0], [9.0, 11.0]]);
        sliced.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_slice_with_negative_step() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        );

        // Test step=-1 along first dimension (reverse rows)
        let sliced = tensor.clone().slice([s![0..3;-1]]);
        let expected = TensorData::from([
            [9.0, 10.0, 11.0, 12.0],
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 2.0, 3.0, 4.0],
        ]);
        sliced.into_data().assert_eq(&expected, false);

        // Test step=-1 along second dimension (reverse columns)
        let sliced = tensor.clone().slice(s![.., 0..4;-1]);
        let expected = TensorData::from([
            [4.0, 3.0, 2.0, 1.0],
            [8.0, 7.0, 6.0, 5.0],
            [12.0, 11.0, 10.0, 9.0],
        ]);
        sliced.into_data().assert_eq(&expected, false);

        // Test step=-2 along first dimension
        let sliced = tensor.clone().slice([s![0..3;-2]]);
        let expected = TensorData::from([[9.0, 10.0, 11.0, 12.0], [1.0, 2.0, 3.0, 4.0]]);
        sliced.into_data().assert_eq(&expected, false);

        // Test step=-2 along second dimension
        let sliced = tensor.clone().slice(s![.., 0..4;-2]);
        let expected = TensorData::from([[4.0, 2.0], [8.0, 6.0], [12.0, 10.0]]);
        sliced.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_slice_with_mixed_steps() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        );

        // Test positive step along first dimension, negative along second
        let sliced = tensor.clone().slice(s![0..3;2, 0..4;-1]);
        let expected = TensorData::from([[4.0, 3.0, 2.0, 1.0], [12.0, 11.0, 10.0, 9.0]]);
        sliced.into_data().assert_eq(&expected, false);

        // Test negative step along first dimension, positive along second
        let sliced = tensor.clone().slice(s![0..3;-1, 0..4;2]);
        let expected = TensorData::from([[9.0, 11.0], [5.0, 7.0], [1.0, 3.0]]);
        sliced.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_slice_with_steps_1d() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &device,
        );

        // Test positive step
        let sliced = tensor.clone().slice([s![0..10;2]]);
        let expected = TensorData::from([1.0, 3.0, 5.0, 7.0, 9.0]);
        sliced.into_data().assert_eq(&expected, false);

        // Test negative step
        let sliced = tensor.clone().slice([s![0..10;-1]]);
        let expected = TensorData::from([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        sliced.into_data().assert_eq(&expected, false);

        // Test negative step with partial range
        let sliced = tensor.clone().slice([s![2..8;-2]]);
        let expected = TensorData::from([8.0, 6.0, 4.0]);
        sliced.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_slice_with_steps_3d() {
        let device = Default::default();
        let tensor = TestTensor::<3>::from_data(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ],
            &device,
        );

        // Test step=2 along first dimension
        let sliced = tensor.clone().slice(s![0..4;2, .., ..]);
        let expected = TensorData::from([[[1.0, 2.0], [3.0, 4.0]], [[9.0, 10.0], [11.0, 12.0]]]);
        sliced.into_data().assert_eq(&expected, false);

        // Test step=-1 along all dimensions
        let sliced = tensor.clone().slice(s![0..4;-1, 0..2;-1, 0..2;-1]);
        let expected = TensorData::from([
            [[16.0, 15.0], [14.0, 13.0]],
            [[12.0, 11.0], [10.0, 9.0]],
            [[8.0, 7.0], [6.0, 5.0]],
            [[4.0, 3.0], [2.0, 1.0]],
        ]);
        sliced.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_slice_int_tensor_with_steps() {
        let device = Default::default();
        let tensor =
            TestTensorInt::<2>::from_data([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], &device);

        // Test step=2 along first dimension
        let sliced = tensor.clone().slice([s![0..3;2]]);
        let expected = TensorData::from([[1i32, 2, 3, 4], [9, 10, 11, 12]]);
        sliced.into_data().assert_eq(&expected, false);

        // Test step=-1 along second dimension
        let sliced = tensor.clone().slice(s![.., 0..4;-1]);
        let expected = TensorData::from([[4i32, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]]);
        sliced.into_data().assert_eq(&expected, false);
    }
}

#[burn_tensor_testgen::testgen(slice)]
mod tests {
    use super::*;
    use burn_tensor::{Int, Slice, Tensor, TensorData, as_type, s};

    #[test]
    fn should_support_const_and_full() {
        static SLICES: [Slice; 2] = [Slice::full(), Slice::new(2, None, 1)];
        assert_eq!(SLICES[0], Slice::new(0, None, 1));
        assert_eq!(SLICES[1], Slice::new(2, None, 1));
    }

    #[test]
    fn should_support_default() {
        assert_eq!(Slice::default(), Slice::new(0, None, 1));
    }

    #[test]
    fn should_support_copy() {
        let mut slice = Slice::new(1, Some(3), 2);
        let slice_copy = slice;

        slice.end = Some(4);

        assert_eq!(slice, Slice::new(1, Some(4), 2));
        assert_eq!(slice_copy, Slice::new(1, Some(3), 2));
    }

    #[test]
    fn should_support_slice_dim_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        // Test with range (negative index)
        let output = tensor.clone().slice_dim(0, -2..);
        output
            .into_data()
            .assert_eq(&TensorData::from([1.0, 2.0]), false);

        // Test with Slice directly
        let slice = Slice::new(1, None, 1); // equivalent to 1..
        let output = tensor.slice_dim(0, slice);
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
    fn should_support_slice_dim_with_step() {
        let data = TensorData::from([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]);
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        // Test 1: Slice dimension 1 with step=2 using s! macro
        let output = tensor.clone().slice_dim(1, s![0..4;2]);
        output
            .into_data()
            .assert_eq(&TensorData::from([[0.0, 2.0], [4.0, 6.0]]), false);

        // Test 2: Slice dimension 1 with step=2 using Slice directly
        let slice = Slice::new(0, Some(4), 2);
        let output = tensor.slice_dim(1, slice);
        output
            .into_data()
            .assert_eq(&TensorData::from([[0.0, 2.0], [4.0, 6.0]]), false);
    }

    #[test]
    fn should_support_slice_dim_with_negative_step() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        // Slice dimension 1 with negative step (reverse columns)
        let output = tensor.slice_dim(1, s![..;-1]);
        output
            .into_data()
            .assert_eq(&TensorData::from([[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]]), false);
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
    fn should_support_slice_fill_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);

        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.slice_fill([0..2], -1.0);
        let expected = TensorData::from([-1.0, -1.0, 2.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_fill_cast_f32() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device).cast(burn_tensor::DType::F32);

        tensor.slice_fill(s![0..2], 1.0)
            .into_data()
            .assert_eq(&TensorData::from([1.0, 1.0, 2.0]), false);
    }

    #[test]
    fn should_support_slice_fill_cast_f64() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device).cast(burn_tensor::DType::F64);

        tensor.slice_fill(s![0..2], 1.0)
            .into_data()
            .assert_eq(&TensorData::from([1.0, 1.0, 2.0]), false);
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
    fn should_support_slice_fill_with_positive_step() {
        let device = Default::default();

        // Test 1D tensor with step
        let tensor = TestTensor::<1>::zeros([10], &device);
        let output = tensor.slice_fill(s![0..10;2], 5.0);
        let expected = TensorData::from([5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0]);
        output.into_data().assert_eq(&expected, false);

        // Test 2D tensor with step on first dimension
        let tensor = TestTensor::<2>::zeros([4, 4], &device);
        let output = tensor.slice_fill(s![0..4;2, ..], 3.0);
        let expected = TensorData::from([
            [3.0, 3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);
        output.into_data().assert_eq(&expected, false);

        // Test 2D tensor with step on second dimension
        let tensor = TestTensor::<2>::zeros([3, 6], &device);
        let output = tensor.slice_fill(s![.., 0..6;3], 2.0);
        let expected = TensorData::from([
            [2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_fill_with_negative_step() {
        let device = Default::default();

        // Test 1D tensor with negative step (reverse fill)
        let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0, 5.0], &device);
        let output = tensor.slice_fill(s![0..5;-1], 10.0);
        // Should reverse the indices [4,3,2,1,0] and fill them with 10.0
        let expected = TensorData::from([10.0, 10.0, 10.0, 10.0, 10.0]);
        output.into_data().assert_eq(&expected, false);

        // Test 2D tensor with negative step
        let tensor = TestTensor::<2>::from_data(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            &device,
        );
        let output = tensor.slice_fill(s![.., 0..3;-2], -1.0);
        // Should fill columns in reverse order with step 2: indices 2, 0
        let expected = TensorData::from([[-1.0, 2.0, -1.0], [-1.0, 5.0, -1.0], [-1.0, 8.0, -1.0]]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_fill_with_mixed_steps() {
        let device = Default::default();

        // Test 2D tensor with mixed positive and negative steps
        let tensor = TestTensor::<2>::zeros([4, 6], &device);
        let output = tensor.slice_fill(s![0..4;2, 0..6;-3], 7.0);
        // Step 2 on dim 0 selects rows 0, 2
        // Step -3 on dim 1 with range 0..6 reverses and takes every 3rd: indices [5, 2]
        let expected = TensorData::from([
            [0.0, 0.0, 7.0, 0.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 0.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        output.into_data().assert_eq(&expected, false);

        // Test 3D tensor with steps
        let tensor = TestTensor::<3>::zeros([2, 4, 4], &device);
        let output = tensor.slice_fill(s![.., 0..4;2, 0..4;-2], 1.0);
        // Step 2 on dim 1 selects rows 0, 2
        // Step -2 on dim 2 with range 0..4 reverses and takes every 2nd: indices [3, 1]
        let expected_slice = [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let expected = TensorData::from([expected_slice.clone(), expected_slice]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn slice_should_not_corrupt_potentially_inplace_operations() {
        let tensor = TestTensorInt::<1>::from_data([1, 2, 3, 4, 5], &Default::default());
        let tensor = tensor.clone().slice([0..3]) + tensor.clone().slice([2..5]);

        let expected = TensorData::from([4, 6, 8]);

        tensor.into_data().assert_eq(&expected, false);
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

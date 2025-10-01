// Import the shared macro
use crate::include_models;
include_models!(
    slice,
    slice_shape,
    slice_scalar,
    slice_mixed,
    slice_shape_gather,
    slice_shape_runtime,
    slice_shape_multi,
    slice_shape_negative,
    slice_shape_negative_range,
    slice_1d_tensor,
    slice_shape_start_tensor_end,
    slice_tensor_start_shape_end,
    slice_axes,
    slice_with_steps,
    slice_shape_with_steps
);

#[cfg(test)]
mod tests {

    use alloc::vec;

    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn slice() {
        let model: slice::Model<TestBackend> = slice::Model::default();
        let device = Default::default();

        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                [11., 12., 13., 14., 15., 16., 17., 18., 19., 20.],
                [21., 22., 23., 24., 25., 26., 27., 28., 29., 30.],
                [31., 32., 33., 34., 35., 36., 37., 38., 39., 40.],
                [41., 42., 43., 44., 45., 46., 47., 48., 49., 50.],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [1f32, 2., 3., 4., 5.],
            [11f32, 12., 13., 14., 15.],
            [21., 22., 23., 24., 25.],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn slice_shape() {
        let model: slice_shape::Model<TestBackend> = slice_shape::Model::default();
        let device = Default::default();

        let input = Tensor::<TestBackend, 4>::zeros([1, 2, 3, 1], &device);

        // Slice Start == 1, End == 3
        let output = model.forward(input);

        assert_eq!(output, [2, 3]);
    }

    #[test]
    fn slice_scalar() {
        let model: slice_scalar::Model<TestBackend> = slice_scalar::Model::default();
        let device = Default::default();

        let input = Tensor::<TestBackend, 2>::ones([5, 3], &device);
        let start = 1;
        let end = 4;

        let output = model.forward(input, start, end);

        let expected_shape = [3, 3];
        assert_eq!(output.shape().dims, expected_shape);
    }

    #[test]
    fn slice_mixed() {
        let model: slice_mixed::Model<TestBackend> = slice_mixed::Model::default();
        let device = Default::default();

        // Create test input tensor [5, 3]
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ],
            &device,
        );

        // Test case: slice from index 1 to 4 (so [1:4, :])
        let end: i64 = 4;
        let output = model.forward(input, end);

        // Expected: input[1:4, :] should give us rows 1, 2, 3 (3 rows total)
        let expected = TensorData::from([[4.0f32, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn slice_shape_gather() {
        let model: slice_shape_gather::Model<TestBackend> = slice_shape_gather::Model::default();
        let device = Default::default();

        // Create test input tensor [2, 4, 6, 8]
        let input = Tensor::<TestBackend, 4>::ones([2, 4, 6, 8], &device);

        let output = model.forward(input.clone());

        // The graph does: Shape -> Gather(axis=0, indices=[1]) -> Slice(ends=gathered_dim)
        // Shape produces [2, 4, 6, 8]
        // Gather with index 1 produces 4 (scalar)
        // Slice uses starts=[0], ends=4, axes=[1], steps=[1]
        // So it slices axis 1 from 0:4, which is the full dimension
        // Result should be same shape as input: [2, 4, 6, 8]

        assert_eq!(output.shape().dims, [2, 4, 6, 8]);

        // Since we're slicing the full dimension (0:4), output should equal input
        let input_data = input.to_data();
        let output_data = output.to_data();
        input_data.assert_eq(&output_data, true);
    }

    #[test]
    fn slice_shape_runtime() {
        let model: slice_shape_runtime::Model<TestBackend> = slice_shape_runtime::Model::default();
        let device = Default::default();

        // Create test input tensor [10, 8, 6]
        let input = Tensor::<TestBackend, 3>::ones([10, 8, 6], &device);

        // Create shape input tensor [3, 4] - its shape will be used as slice ends
        let shape_input = Tensor::<TestBackend, 2>::ones([3, 4], &device);

        let output = model.forward(input, shape_input);

        // The graph extracts shape [3, 4] and uses it as ends for slicing
        // Slice uses starts=[0, 0], ends=[3, 4], axes=[0, 1]
        // So it slices first two dimensions: [0:3, 0:4, :]
        // Result shape should be [3, 4, 6]
        assert_eq!(output.shape().dims, [3, 4, 6]);
    }

    #[test]
    fn slice_shape_multi() {
        let model: slice_shape_multi::Model<TestBackend> = slice_shape_multi::Model::default();
        let device = Default::default();

        // Create test input tensor [8, 6, 10, 12]
        let input = Tensor::<TestBackend, 4>::ones([8, 6, 10, 12], &device);

        // Create shape tensors whose shapes will be used as slice parameters
        // start_shape_input has shape [1, 2, 3] -> used as start indices
        let start_shape_input = Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device);

        // end_shape_input has shape [5, 4, 7] -> used as end indices
        let end_shape_input = Tensor::<TestBackend, 3>::zeros([5, 4, 7], &device);

        let output = model.forward(input, start_shape_input, end_shape_input);

        // The graph extracts shapes and uses them for slicing
        // Slice uses starts=[1, 2, 3], ends=[5, 4, 7], axes=[0, 1, 2]
        // So it slices: [1:5, 2:4, 3:7, :]
        // Result shape should be [4, 2, 4, 12]
        assert_eq!(output.shape().dims, [4, 2, 4, 12]);
    }

    #[test]
    fn slice_shape_negative() {
        let model: slice_shape_negative::Model<TestBackend> =
            slice_shape_negative::Model::default();
        let device = Default::default();

        // Create test input tensor [2, 3, 4, 5]
        let input = Tensor::<TestBackend, 4>::ones([2, 3, 4, 5], &device);

        let output = model.forward(input);

        // The graph does: Shape -> Slice(starts=[-1], ends=[INT64_MAX])
        // Shape produces [2, 3, 4, 5]
        // Slice with [-1:] should get the last element: 5
        assert_eq!(output, [5]);
    }

    #[test]
    fn slice_shape_negative_range() {
        let model: slice_shape_negative_range::Model<TestBackend> =
            slice_shape_negative_range::Model::default();
        let device = Default::default();

        // Create test input tensor [2, 3, 4, 5]
        let input = Tensor::<TestBackend, 4>::ones([2, 3, 4, 5], &device);

        let output: [i64; 2] = model.forward(input);

        // The graph does: Shape -> Slice(starts=[-3], ends=[-1])
        // Shape produces [2, 3, 4, 5]
        // Slice with [-3:-1] should get elements from 3rd last to 2nd last: [3, 4]
        assert_eq!(output, [3i64, 4i64]);
    }

    #[test]
    fn slice_1d_tensor() {
        let model: slice_1d_tensor::Model<TestBackend> = slice_1d_tensor::Model::default();
        let device = Default::default();

        // Create test input tensor [4, 5, 6] using range and reshape
        let input = Tensor::<TestBackend, 1, burn::tensor::Int>::arange(1..121, &device)
            .float()
            .reshape([4, 5, 6]);

        // Create 1D tensors for starts and ends
        let starts = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints([1i64, 2i64], &device);
        let ends = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints([3i64, 5i64], &device);

        let output = model.forward(input, starts, ends);

        // Expected: input[1:3, 2:5, :] -> shape [2, 3, 6]
        let expected_shape = [2, 3, 6];
        assert_eq!(output.shape().dims, expected_shape);

        // Verify the data is correctly sliced
        // Expected values: slicing [1:3, 2:5, :] from the reshaped tensor
        // Create expected tensor directly using from_floats
        let expected_data: alloc::vec::Vec<f32> =
            (43..61).chain(73..91).map(|x| x as f32).collect();
        let expected = Tensor::<TestBackend, 1>::from_floats(expected_data.as_slice(), &device)
            .reshape([2, 3, 6]);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn slice_shape_start_tensor_end() {
        let model: slice_shape_start_tensor_end::Model<TestBackend> =
            slice_shape_start_tensor_end::Model::default();
        let device = Default::default();

        // Create test input tensor [6, 8, 10]
        let input = Tensor::<TestBackend, 3>::ones([6, 8, 10], &device);

        // Create shape input tensor [2, 3] - its shape will be used as starts
        let shape_input = Tensor::<TestBackend, 2>::ones([2, 3], &device);

        // Create 1D tensor for ends
        let ends = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints([5i64, 8i64], &device);

        let output = model.forward(input, shape_input, ends);

        // The graph extracts shape [2, 3] and uses it as starts
        // Slice uses starts=[2, 3], ends=[5, 8], axes=[0, 1]
        // So it slices: [2:5, 3:8, :]
        // Result shape should be [3, 5, 10]
        assert_eq!(output.shape().dims, [3, 5, 10]);
    }

    #[test]
    fn slice_tensor_start_shape_end() {
        let model: slice_tensor_start_shape_end::Model<TestBackend> =
            slice_tensor_start_shape_end::Model::default();
        let device = Default::default();

        // Create test input tensor [10, 12, 8]
        let input = Tensor::<TestBackend, 3>::ones([10, 12, 8], &device);

        // Create 1D tensor for starts
        let starts = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints([2i64, 3i64], &device);

        // Create shape input tensor [6, 10] - its shape will be used as ends
        let shape_input = Tensor::<TestBackend, 2>::ones([6, 10], &device);

        let output = model.forward(input, starts, shape_input);

        // The graph extracts shape [6, 10] and uses it as ends
        // Slice uses starts=[2, 3], ends=[6, 10], axes=[0, 1]
        // So it slices: [2:6, 3:10, :]
        // Result shape should be [4, 7, 8]
        assert_eq!(output.shape().dims, [4, 7, 8]);
    }

    #[test]
    fn slice_with_steps() {
        let model: slice_with_steps::Model<TestBackend> = slice_with_steps::Model::default();
        let device = Default::default();

        // Create test input with shape [10, 10, 12]
        let mut data = vec![];
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..12 {
                    data.push((i * 100 + j * 10 + k) as f32);
                }
            }
        }

        let input = Tensor::<TestBackend, 1>::from_data(TensorData::from(data.as_slice()), &device)
            .reshape([10, 10, 12]);

        let output = model.forward(input);

        // Expected output shape: [5, 3, 12]
        // dim0: indices 0,2,4,6,8 (step=2)
        // dim1: indices 1,4,7 (step=3)
        // dim2: all 12 elements but reversed (step=-1)
        assert_eq!(output.shape().dims, [5, 3, 12]);

        // Verify some values
        let output_data = output.to_data();
        let values = output_data.to_vec::<f32>().unwrap();

        // First element should be from [0, 1, 11] (reversed last dim)
        // [0, 1, 11] -> 0*100 + 1*10 + 11 = 21
        assert_eq!(values[0], 21.0);
        // Element at [0, 1, 0] should be from [0, 4, 11]
        // [0, 4, 11] -> 0*100 + 4*10 + 11 = 51
        assert_eq!(values[12], 51.0);
        // Element at [1, 0, 0] should be from [2, 1, 11]
        // [2, 1, 11] -> 2*100 + 1*10 + 11 = 221
        assert_eq!(values[36], 221.0);
    }

    #[test]
    fn slice_shape_with_steps() {
        let model: slice_shape_with_steps::Model<TestBackend> =
            slice_shape_with_steps::Model::default();
        let device = Default::default();

        // Create test input with shape [2, 3, 4, 5, 6, 7]
        let input = Tensor::<TestBackend, 6>::ones([2, 3, 4, 5, 6, 7], &device);

        let output = model.forward(input);

        // The model should extract shape [2, 3, 4, 5, 6, 7]
        // Then slice it with [0:6:2] to get [2, 4, 6]
        assert_eq!(output, [2i64, 4, 6]);
    }

    #[test]
    fn slice_axes() {
        // This test validates that the axes parameter is correctly handled.
        // Without axes, slice would operate on dimension 0.
        // With axes=[1], slice operates on dimension 1.
        let model: slice_axes::Model<TestBackend> = slice_axes::Model::default();
        let device = Default::default();

        // Create test input with shape [2, 4, 6]
        let test_input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                    [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                ],
                [
                    [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                    [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
                    [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
                    [43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
                ],
            ],
            &device,
        );

        let output = model.forward(test_input.clone());

        // Expected output shape: [2, 2, 6]
        // The model slices axis 1 from index 1 to 3
        // This means taking rows 1 and 2 from each batch
        assert_eq!(output.shape().dims, [2, 2, 6], "Output shape mismatch");

        // Expected output: input[:, 1:3, :]
        let expected = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                ],
                [
                    [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
                    [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
                ],
            ],
            &device,
        );

        output.to_data().assert_eq(&expected.to_data(), true);
    }
}

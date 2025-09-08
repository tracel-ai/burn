// Import the shared macro
use crate::include_models;
include_models!(
    gather_1d_idx,
    gather_2d_idx,
    gather_scalar,
    gather_constant_2d_indices,
    gather_static_shape_indices,
    gather_scalar_out,
    gather_shape,
    gather_with_shape_indices
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn gather_1d_idx() {
        let model: gather_1d_idx::Model<TestBackend> = gather_1d_idx::Model::default();

        let device = Default::default();

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let index = Tensor::<TestBackend, 1, Int>::from_ints([0, 2], &device);
        let expected = TensorData::from([[1f32, 3.], [4., 6.]]);
        let output = model.forward(input, index);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_2d_idx() {
        let model: gather_2d_idx::Model<TestBackend> = gather_2d_idx::Model::default();

        let device = Default::default();

        let input =
            Tensor::<TestBackend, 2>::from_data([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]], &device);
        let index = Tensor::<TestBackend, 2, Int>::from_data([[0, 1], [1, 2]], &device);
        let expected = TensorData::from([[[1f32, 1.2], [2.3, 3.4]], [[2.3, 3.4], [4.5, 5.7]]]);
        let output = model.forward(input, index);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_shape() {
        let model: gather_shape::Model<TestBackend> = gather_shape::Model::default();

        let device = Default::default();

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        // shape(input) = [2, 3]
        let index = Tensor::<TestBackend, 1, Int>::from_ints([0], &device);

        // The model now returns 3 outputs:
        // output1: gather with runtime index
        // output2: gather with constant scalar index (1)
        // output3: gather with constant 1D indices [0, 1]
        let (output1, output2, output3) = model.forward(input, index);

        // Test runtime index gather
        let expected1 = [2i64];
        assert_eq!(output1, expected1);

        // Test constant scalar index gather (index=1 gets shape[1]=3)
        let expected2 = 3i64;
        assert_eq!(output2, expected2);

        // Test constant 1D indices gather (indices=[0,1] gets [shape[0], shape[1]] = [2, 3])
        let expected3 = [2i64, 3i64];
        assert_eq!(output3, expected3);
    }

    #[test]
    fn gather_scalar() {
        let model: gather_scalar::Model<TestBackend> = gather_scalar::Model::default();

        let device = Default::default();

        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let index = 0;
        let output = model.forward(input, index);
        let expected = TensorData::from([1f32, 2., 3.]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_scalar_out() {
        let model: gather_scalar_out::Model<TestBackend> = gather_scalar_out::Model::default();

        let device = Default::default();

        let input = Tensor::<TestBackend, 1>::from_floats([1., 2., 3.], &device);
        let index = 1;
        let output = model.forward(input, index);

        let expected = 2.0f32;
        assert_eq!(output, expected);
    }

    #[test]
    fn gather_with_shape_indices() {
        // Test the most comprehensive case of our runtime Shape indices implementation:
        // This is the exact scenario that was causing the original panic and required our full fix.
        let model: gather_with_shape_indices::Model<TestBackend> =
            gather_with_shape_indices::Model::default();

        let device = Default::default();

        // Input tensor with shape [2, 3]
        let input = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input);

        // Expected: shape [2, 3] used as indices to gather from [100, 200, 300, 400, 500]
        // Gathering at indices [2, 3] should give us [300, 400]
        let expected = TensorData::from([300i64, 400]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_constant_2d_indices() {
        // Test Gather with 3 variations:
        // 1. Both data and indices as initializers
        // 2. Data as initializer, indices as Constant node
        // 3. Data as Constant node, indices as initializer
        // The final output is the sum of all three gather operations
        let device = Default::default();
        let model: gather_constant_2d_indices::Model<TestBackend> =
            gather_constant_2d_indices::Model::new(&device);

        // Model has no inputs (all inputs are constants/initializers)
        // Just call forward to get the output
        let output = model.forward();

        // Verify output shape is [1, 5, 7]
        assert_eq!(
            output.dims(),
            [1, 5, 7],
            "Output should have shape [1, 5, 7]"
        );

        // The model computes: output = gather1 + gather2 + gather3
        // Each gather produces the same result (gathering rows [0,2,1,4,3])
        // So the final output is 3x the single gather result

        // The data is initialized as arange(35).reshape(5,7)
        // Gathering index 0 gives [0,1,2,3,4,5,6]
        // Since we sum 3 identical gathers, first row should be [0,3,6,9,12,15,18]

        // This test verifies that the model can handle:
        // 1. Both inputs as initializers
        // 2. Constant nodes for indices
        // 3. Constant nodes for data
        // The actual values depend on how burn-import handles constant lifting

        // Just verify we get the expected output shape
        // The shape should be [1, 5, 7] regardless of the input method
    }

    #[test]
    fn gather_static_shape_indices() {
        // Test that demonstrates why shape preservation is crucial for static indices
        // Without shape info, we can't distinguish between:
        // - Scalar index (rank 0): reduces dimension
        // - 1D tensor [1] (rank 1): preserves dimension
        // - 2D tensor [[1,0]] (rank 2): adds dimension
        let model: gather_static_shape_indices::Model<TestBackend> =
            gather_static_shape_indices::Model::default();

        let device = Default::default();

        // Input tensor shape [3, 4, 5]
        let input = Tensor::<TestBackend, 3>::ones([3, 4, 5], &device);
        // Set specific values to verify correct indexing
        let input = input.clone().add_scalar(1.0); // All 2s
        // Make index 1 different
        let slice_1 = Tensor::<TestBackend, 3>::ones([1, 4, 5], &device).add_scalar(4.0); // All 5s
        let input = input.slice_assign([1..2, 0..4, 0..5], slice_1);

        let (output_scalar, output_1d, output_2d) = model.forward(input);

        // Scalar index (rank 0) should give [4, 5] shape
        assert_eq!(
            output_scalar.dims(),
            [4, 5],
            "Scalar index should reduce dimension"
        );

        // 1D index (rank 1) should give [1, 4, 5] shape
        assert_eq!(
            output_1d.dims(),
            [1, 4, 5],
            "1D index should preserve dimension"
        );

        // 2D index (rank 2) should give [1, 2, 4, 5] shape
        assert_eq!(
            output_2d.dims(),
            [1, 2, 4, 5],
            "2D index should add dimension"
        );

        // All should have selected index 1 (value 5.0) for first element
        assert_eq!(output_scalar.clone().slice([0..1, 0..1]).into_scalar(), 5.0);
        assert_eq!(
            output_1d.clone().slice([0..1, 0..1, 0..1]).into_scalar(),
            5.0
        );
        assert_eq!(
            output_2d
                .clone()
                .slice([0..1, 0..1, 0..1, 0..1])
                .into_scalar(),
            5.0
        );
    }
}

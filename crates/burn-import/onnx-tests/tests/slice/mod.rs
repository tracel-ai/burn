// Import the shared macro
use crate::include_models;
include_models!(slice, slice_shape, slice_scalar, slice_mixed, slice_shape_gather);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn slice() {
        let model: slice::Model<Backend> = slice::Model::default();
        let device = Default::default();

        let input = Tensor::<Backend, 2>::from_floats(
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
        let model: slice_shape::Model<Backend> = slice_shape::Model::default();
        let device = Default::default();

        let input = Tensor::<Backend, 4>::zeros([1, 2, 3, 1], &device);

        // Slice Start == 1, End == 3
        let output = model.forward(input);

        assert_eq!(output, [2, 3]);
    }

    #[test]
    fn slice_scalar() {
        let model: slice_scalar::Model<Backend> = slice_scalar::Model::default();
        let device = Default::default();

        let input = Tensor::<Backend, 2>::ones([5, 3], &device);
        let start = 1;
        let end = 4;

        let output = model.forward(input, start, end);

        let expected_shape = [3, 3];
        assert_eq!(output.shape().dims, expected_shape);
    }

    #[test]
    fn slice_mixed() {
        let model: slice_mixed::Model<Backend> = slice_mixed::Model::default();
        let device = Default::default();

        // Create test input tensor [5, 3]
        let input = Tensor::<Backend, 2>::from_floats(
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
        let expected = TensorData::from([
            [4.0f32, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    #[ignore = "TODO: Fix gather constant handling in ONNX import"]
    fn slice_shape_gather() {
        // This test is currently ignored because the ONNX import doesn't properly handle
        // the gather_indices constant. The generated code tries to reference 'gather_indices'
        // as a variable instead of using the constant value.
        
        let model: slice_shape_gather::Model<Backend> = slice_shape_gather::Model::default();
        let device = Default::default();

        // Create test input tensor [2, 4, 6, 8]
        let input = Tensor::<Backend, 4>::ones([2, 4, 6, 8], &device);

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
}

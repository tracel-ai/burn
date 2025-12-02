use crate::include_models;
include_models!(
    unsqueeze_like,
    unsqueeze_runtime_axes,
    unsqueeze_int_to_shape,
    squeeze_unsqueeze_roundtrip
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};

    use crate::backend::TestBackend;

    #[test]
    fn unsqueeze_runtime_axes() {
        let device = Default::default();
        let model: unsqueeze_runtime_axes::Model<TestBackend> =
            unsqueeze_runtime_axes::Model::new(&device);
        let input_shape = Shape::from([3, 4, 5]);
        let expected_shape = Shape::from([1, 3, 1, 4, 5, 1]);
        let input = Tensor::ones(input_shape, &device);

        // Note: The axes tensor must have rank 1 with a single element
        // as the generated ONNX requires a 1D tensor for static shape operations
        // see unsqueeze.onnx
        let axes = Tensor::from_ints([2], &device);
        let output = model.forward(input, axes);
        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn unsqueeze_like() {
        let device = Default::default();
        let model = unsqueeze_like::Model::<TestBackend>::new(&device);
        let input_shape = Shape::from([3, 4, 5]);
        let expected_shape = Shape::from([3, 4, 5, 1]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input, 1.0);
        assert_eq!(expected_shape, output.0.shape());
        assert_eq!(Shape::from([1]), output.1.shape());
    }

    #[test]
    fn unsqueeze_int_to_shape() {
        // Test the direct conversion of Int scalar to Shape array [i64; 1]
        // This demonstrates the optimization where Int scalars are unsqueezed to Shape types
        // rather than tensors, which is crucial for efficient dynamic shape operations
        // The generated model takes an i64 scalar and returns a Shape array [i64; 1]
        let device = Default::default();
        let model = unsqueeze_int_to_shape::Model::<TestBackend>::new(&device);

        // Input: scalar int64 value
        let scalar_value = 42i64;

        // Expected output: Shape array [i64; 1] containing the same value
        let output_shape = model.forward(scalar_value);

        // Verify the output is a Shape array with our input value
        assert_eq!(output_shape[0], scalar_value);

        // This shows that the optimization is working:
        // The Int scalar is directly converted to a Shape array without tensor allocation
    }

    #[test]
    fn squeeze_unsqueeze_roundtrip() {
        // Test the complete roundtrip: Tensor<1> -> squeeze -> Scalar -> unsqueeze -> Tensor<1>
        // This verifies that the squeeze/unsqueeze operations are symmetric and maintain
        // type consistency for shape manipulation patterns common in ONNX models
        let device = Default::default();
        let model = squeeze_unsqueeze_roundtrip::Model::<TestBackend>::new(&device);

        // Input: 1D tensor with a value
        let input_value = 256i64;
        let input_tensor =
            Tensor::<TestBackend, 1, burn::tensor::Int>::from_data([input_value], &device);

        // The roundtrip should preserve the value
        // Note: The output is a Shape type [i64; 1], not a Tensor
        let output_shape = model.forward(input_tensor.clone());

        // Verify the value is preserved through the squeeze/unsqueeze roundtrip
        assert_eq!(output_shape[0], input_value);
    }
}

// Import the shared macro
use crate::include_models;
include_models!(
    constant_f32,
    constant_f64,
    constant_i32,
    constant_i64,
    constant_bool,
    constant_shape,
    constant_tensor_f32,
    constant_tensor_i32,
    constant_tensor_bool,
    constant_empty_tensor_f32,
    rank_inference_propagation,
    shape_binary_ops_with_constant
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Shape, Tensor};

    use crate::backend::TestBackend;

    #[test]
    fn add_constant_f32() {
        let device = Default::default();
        let model = constant_f32::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<TestBackend, 3>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_constant_f64() {
        let device = Default::default();
        let model = constant_f64::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<TestBackend, 3>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_constant_i32() {
        let device = Default::default();
        let model = constant_i32::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3, Int>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<TestBackend, 3, Int>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_constant_i64() {
        let device = Default::default();
        let model = constant_i64::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3, Int>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<TestBackend, 3, Int>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn or_constant_bool() {
        // Test scalar boolean constant with OR operation
        let device = Default::default();
        let model = constant_bool::Model::<TestBackend>::new(&device);

        // Test with false input - should return true (false OR true = true)
        let output_false = model.forward(false);
        assert_eq!(output_false, true, "false OR true should be true");

        // Test with true input - should return true (true OR true = true)
        let output_true = model.forward(true);
        assert_eq!(output_true, true, "true OR true should be true");
    }

    #[test]
    fn constant_tensor_f32_test() {
        // Test that multidimensional f32 tensor constants are properly loaded
        let device = Default::default();
        let model: constant_tensor_f32::Model<TestBackend> = constant_tensor_f32::Model::default();

        // Create input tensor [2, 3] with values [[1, 2, 3], [4, 5, 6]]
        let input =
            Tensor::<TestBackend, 2>::from_data([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

        // Expected: input + constant where constant is [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]
        // Result: [[2.5, 4.5, 6.5], [8.5, 10.5, 12.5]]
        let expected =
            Tensor::<TestBackend, 2>::from_data([[2.5f32, 4.5, 6.5], [8.5, 10.5, 12.5]], &device)
                .to_data();

        let output = model.forward(input);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn constant_tensor_i32_test() {
        // Test that multidimensional i32 tensor constants are properly loaded with correct dtype
        let device = Default::default();
        let model: constant_tensor_i32::Model<TestBackend> = constant_tensor_i32::Model::default();

        // Create input tensor [2, 3] with values [[1, 2, 3], [4, 5, 6]]
        let input = Tensor::<TestBackend, 2, Int>::from_ints([[1i32, 2, 3], [4, 5, 6]], &device);

        // Expected: input + constant where constant is [[10, 20, 30], [40, 50, 60]]
        // Result: [[11, 22, 33], [44, 55, 66]]
        let expected =
            Tensor::<TestBackend, 2, Int>::from_ints([[11i32, 22, 33], [44, 55, 66]], &device)
                .to_data();

        let output = model.forward(input);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn constant_tensor_bool_test() {
        // Test that multidimensional bool tensor constants are properly loaded
        let device = Default::default();
        let model: constant_tensor_bool::Model<TestBackend> =
            constant_tensor_bool::Model::default();

        // Create input tensor [2, 3] with bool values
        use burn::tensor::Bool;
        let input_data = [[false, false, false], [true, true, true]];
        let input = Tensor::<TestBackend, 2, Bool>::from_bool(
            burn::tensor::TensorData::from(input_data),
            &device,
        );

        // Expected: input OR constant where constant is [[true, false, true], [false, true, false]]
        // Result: [[true, false, true], [true, true, true]]
        let expected_data = [[true, false, true], [true, true, true]];
        let expected = burn::tensor::TensorData::from(expected_data);

        let output = model.forward(input);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn constant_empty_tensor_f32_test() {
        // Test that empty f32 tensor constants with no data are properly handled
        // This tests the fix for the bug where empty float_data and raw_data caused a panic
        let device = Default::default();
        let model: constant_empty_tensor_f32::Model<TestBackend> =
            constant_empty_tensor_f32::Model::default();

        // Create input tensor [2, 3]
        let input =
            Tensor::<TestBackend, 2>::from_data([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

        // The model should load successfully even with an empty tensor constant attribute
        // The output is just the identity of the input
        let output = model.forward(input.clone());

        // Expected: same as input (identity operation)
        let expected = input.to_data();

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn constant_shape() {
        let device = Default::default();
        let model = constant_shape::Model::<TestBackend>::new(&device);

        // Create input tensor with shape [2, 4, 6]
        let input = Tensor::<TestBackend, 3>::zeros(Shape::from([2, 4, 6]), &device);

        // The model tests Shape operations with constants
        // Input shape: [2, 4, 6]
        // Scalar constant: 2
        // Shape constant: [1, 2, 3]
        let (shape_add_scalar, shape_mul_scalar, shape_add_shape, shape_mul_shape) =
            model.forward(input);

        // Check shape_add_scalar: [2, 4, 6] + 2 = [4, 6, 8]
        assert_eq!(shape_add_scalar[0], 4);
        assert_eq!(shape_add_scalar[1], 6);
        assert_eq!(shape_add_scalar[2], 8);

        // Check shape_mul_scalar: [2, 4, 6] * 2 = [4, 8, 12]
        assert_eq!(shape_mul_scalar[0], 4);
        assert_eq!(shape_mul_scalar[1], 8);
        assert_eq!(shape_mul_scalar[2], 12);

        // Check shape_add_shape: [2, 4, 6] + [1, 2, 3] = [3, 6, 9]
        assert_eq!(shape_add_shape[0], 3);
        assert_eq!(shape_add_shape[1], 6);
        assert_eq!(shape_add_shape[2], 9);

        // Check shape_mul_shape: [2, 4, 6] * [1, 2, 3] = [2, 8, 18]
        assert_eq!(shape_mul_shape[0], 2);
        assert_eq!(shape_mul_shape[1], 8);
        assert_eq!(shape_mul_shape[2], 18);
    }

    #[test]
    fn rank_inference_propagation_test() {
        // Regression test for rank inference propagation after Shape type conversions
        // This test ensures that when Constants are converted to Shape types,
        // all downstream nodes get their ranks properly re-inferred.
        // Without the fix, this would fail during import with:
        // "Concat axis 2 is out of bounds for rank 2"

        let device = Default::default();

        // This should succeed with our fix
        // Without the fix, this would panic during model creation with:
        // "Concat axis 2 is out of bounds for rank 2"
        let model = rank_inference_propagation::Model::<TestBackend>::new(&device);

        // Create a 3D input tensor (batch=2, sequence=4, features=384)
        let input = Tensor::<TestBackend, 3>::ones([2, 4, 384], &device);

        // Run the model
        let output = model.forward(input);

        // Assert the output has the expected shape
        // The model:
        // 1. Gets shape of input [2, 4, 384]
        // 2. Slices to get [2, 4]
        // 3. Concatenates with constant [8, 16] to get [2, 4, 8, 16]
        // 4. MatMul operations produce [2, 4, 64] each
        // 5. Concat along axis 2 produces [2, 4, 128]
        // 6. Reshape to [2, 4, 8, 16]

        let dims = output.dims();
        assert_eq!(dims.len(), 4, "Output should be 4D tensor");
        assert_eq!(dims[0], 2, "Batch dimension should be 2");
        assert_eq!(dims[1], 4, "Sequence dimension should be 4");
        assert_eq!(dims[2], 8, "Third dimension should be 8");
        assert_eq!(dims[3], 16, "Fourth dimension should be 16");

        // Verify total number of elements is preserved
        let total_elements: usize = dims.iter().product();
        assert_eq!(
            total_elements,
            2 * 4 * 128,
            "Total elements should match concat output"
        );
    }

    #[test]
    fn shape_binary_ops_with_constant() {
        // Test that constant tensors are properly converted to Shape type
        // when used in binary operations (add/sub/mul/div) with Shape inputs.
        // This specifically tests the propagation case where a Shape type
        // is propagated through the graph and constants need to be converted
        // during propagation (not just during initial conversion).
        //
        // Without the fix, constants wouldn't be converted to Shape during
        // propagation, causing type mismatches in binary operations.

        let device = Default::default();
        let model = shape_binary_ops_with_constant::Model::<TestBackend>::new(&device);

        // Create input tensor with shape [2, 8, 3]
        let input = Tensor::<TestBackend, 3>::ones(Shape::from([2, 8, 3]), &device);

        // Run the model
        let output = model.forward(input);

        // The model performs on shape [2, 8, 3]:
        // 1. Add [10, 20, 30]: [2+10, 8+20, 3+30] = [12, 28, 33]
        // 2. Divide by [2, 2, 2]: [12/2, 28/2, 33/2] = [6, 14, 16]
        // 3. Subtract [3, 4, 5]: [6-3, 14-4, 16-5] = [3, 10, 11]
        // 4. Multiply by [4, 5, 6]: [3*4, 10*5, 11*6] = [12, 50, 66]

        assert_eq!(output[0], 12, "First element should be 12");
        assert_eq!(output[1], 50, "Second element should be 50");
        assert_eq!(output[2], 66, "Third element should be 66");
    }
}

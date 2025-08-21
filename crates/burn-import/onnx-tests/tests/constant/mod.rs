// Import the shared macro
use crate::include_models;
include_models!(
    constant_f32,
    constant_f64,
    constant_i32,
    constant_i64,
    constant_shape,
    rank_inference_propagation
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
}

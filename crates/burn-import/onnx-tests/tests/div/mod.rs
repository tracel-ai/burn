// Import the shared macro
use crate::include_models;
include_models!(div, div_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn div_tensor_by_scalar_and_tensor_by_tensor() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: div::Model<TestBackend> = div::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats([[[[3., 6., 6., 9.]]]], &device);
        let scalar1 = 9.0f64;
        let scalar2 = 3.0f64;
        let output = model.forward(input, scalar1, scalar2);
        let expected = TensorData::from([[[[1f32, 2., 2., 3.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn div_shape_with_scalar_and_shape() {
        // Initialize the model
        let device = Default::default();
        let model: div_shape::Model<TestBackend> = div_shape::Model::new(&device);

        // Create input tensors
        let input1 = Tensor::<TestBackend, 3>::ones([8, 12, 16], &device);
        let input2 = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let (shape_div_scalar, shape_div_shape) = model.forward(input1, input2);

        // Expected outputs
        let expected_scalar = [4, 6, 8]; // shape1 [8, 12, 16] / 2
        let expected_shape = [4, 4, 4]; // shape1 [8, 12, 16] / shape2 [2, 3, 4]

        assert_eq!(shape_div_scalar, expected_scalar);
        assert_eq!(shape_div_shape, expected_shape);
    }
}

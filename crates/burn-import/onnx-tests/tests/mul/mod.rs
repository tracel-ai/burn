// Import the shared macro
use crate::include_models;
include_models!(mul, mul_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn mul_scalar_with_tensor_and_tensor_with_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: mul::Model<TestBackend> = mul::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let scalar = 6.0f64;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[126f32, 252., 378., 504.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn mul_shape_with_scalar_and_shape() {
        // Initialize the model
        let model: mul_shape::Model<TestBackend> = mul_shape::Model::default();

        let device = Default::default();
        // Create input tensors
        let input1 = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let input2 = Tensor::<TestBackend, 3>::ones([1, 2, 3], &device);
        let (shape_times_scalar, shape_times_shape) = model.forward(input1, input2);

        // Expected outputs
        let expected_scalar = [4, 6, 8]; // shape1 [2, 3, 4] * 2
        let expected_shape = [2, 6, 12]; // shape1 [2, 3, 4] * shape2 [1, 2, 3]

        assert_eq!(shape_times_scalar, expected_scalar);
        assert_eq!(shape_times_shape, expected_shape);
    }
}

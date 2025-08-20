// Include the models for this node type
use crate::include_models;
include_models!(add, add_int, add_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn add_scalar_to_tensor_and_tensor_to_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: add::Model<TestBackend> = add::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let scalar = 2f64;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[9f32, 10., 11., 12.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_scalar_to_int_tensor_and_int_tensor_to_int_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: add_int::Model<TestBackend> = add_int::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<TestBackend, 4, Int>::from_ints([[[[1, 2, 3, 4]]]], &device);
        let scalar = 2;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[9i64, 11, 13, 15]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_shape_with_scalar_and_shape() {
        // Initialize the model
        let model: add_shape::Model<TestBackend> = add_shape::Model::default();

        let device = Default::default();
        // Create input tensors
        let input1 = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let input2 = Tensor::<TestBackend, 3>::ones([5, 6, 7], &device);
        let (shape_plus_scalar, shape_plus_shape) = model.forward(input1, input2);

        // Expected outputs
        let expected_scalar = [12, 13, 14]; // shape1 [2, 3, 4] + 10
        let expected_shape = [7, 9, 11]; // shape1 [2, 3, 4] + shape2 [5, 6, 7]

        assert_eq!(shape_plus_scalar, expected_scalar);
        assert_eq!(shape_plus_shape, expected_shape);
    }
}

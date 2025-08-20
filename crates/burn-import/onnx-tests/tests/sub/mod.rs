// Include the models for this node type
use crate::include_models;
include_models!(sub, sub_int, sub_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn sub_scalar_from_tensor_and_tensor_from_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: sub::Model<TestBackend> = sub::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let scalar = 3.0f64;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[-12f32, -13., -14., -15.]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sub_scalar_from_int_tensor_and_int_tensor_from_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let model: sub_int::Model<TestBackend> = sub_int::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<TestBackend, 4, Int>::from_ints([[[[1, 2, 3, 4]]]], &device);
        let scalar = 3;
        let output = model.forward(input, scalar);
        let expected = TensorData::from([[[[-12i64, -12, -12, -12]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn sub_shape_with_scalar_and_shape() {
        // Initialize the model
        let model: sub_shape::Model<TestBackend> = sub_shape::Model::default();

        let device = Default::default();
        // Create input tensors
        let input1 = Tensor::<TestBackend, 3>::ones([10, 8, 6], &device);
        let input2 = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let (shape_minus_scalar, shape_minus_shape) = model.forward(input1, input2);

        // Expected outputs
        let expected_scalar = [9, 7, 5]; // shape1 [10, 8, 6] - 1
        let expected_shape = [8, 5, 2]; // shape1 [10, 8, 6] - shape2 [2, 3, 4]

        assert_eq!(shape_minus_scalar, expected_scalar);
        assert_eq!(shape_minus_shape, expected_shape);
    }
}

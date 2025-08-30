// Include the models for this node type
use crate::include_models;
include_models!(
    add,
    add_int,
    add_shape,
    add_broadcast,
    add_shape_tensor,
    add_argmax_with_shape
);

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

    #[test]
    fn add_broadcast_tensor_ranks() {
        // Initialize the model
        let model: add_broadcast::Model<TestBackend> = add_broadcast::Model::default();

        let device = Default::default();

        // Create test tensors matching the Python script
        let x_3d = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                [
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                ],
            ],
            &device,
        );

        let y_2d = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ],
            &device,
        );

        let a_2d = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.5, 0.5, 0.5, 0.5],
                [1.5, 1.5, 1.5, 1.5],
                [2.5, 2.5, 2.5, 2.5],
            ],
            &device,
        );

        let b_3d = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                [
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                ],
            ],
            &device,
        );

        let (result1, result2) = model.forward(x_3d, y_2d, a_2d, b_3d);

        // Expected outputs from Python
        let expected1 = TensorData::from([
            [
                [2.0f32, 3.0, 4.0, 5.0],
                [7.0, 8.0, 9.0, 10.0],
                [12.0, 13.0, 14.0, 15.0],
            ],
            [
                [14.0, 15.0, 16.0, 17.0],
                [19.0, 20.0, 21.0, 22.0],
                [24.0, 25.0, 26.0, 27.0],
            ],
        ]);

        let expected2 = TensorData::from([
            [
                [1.5f32, 2.5, 3.5, 4.5],
                [6.5, 7.5, 8.5, 9.5],
                [11.5, 12.5, 13.5, 14.5],
            ],
            [
                [13.5, 14.5, 15.5, 16.5],
                [18.5, 19.5, 20.5, 21.5],
                [23.5, 24.5, 25.5, 26.5],
            ],
        ]);

        result1.to_data().assert_eq(&expected1, true);
        result2.to_data().assert_eq(&expected2, true);
    }
}

// Import the shared macro
use crate::include_models;
include_models!(mul, mul_shape, mul_broadcast, mul_shape_tensor);

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

    #[test]
    fn mul_broadcast_tensor_ranks() {
        let model: mul_broadcast::Model<TestBackend> = mul_broadcast::Model::default();
        let device = Default::default();

        let x_3d = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                [
                    [2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0],
                ],
            ],
            &device,
        );

        let y_2d = Tensor::<TestBackend, 2>::from_floats(
            [
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            &device,
        );

        let a_2d = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        );

        let b_3d = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5, 0.5],
                ],
                [
                    [3.0, 3.0, 3.0, 3.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ],
            &device,
        );

        let (result1, result2) = model.forward(x_3d, y_2d, a_2d, b_3d);

        // Expected outputs from Python evaluation
        let expected1 = TensorData::from([
            [
                [2.0f32, 4.0, 6.0, 8.0],
                [15.0, 18.0, 21.0, 24.0],
                [36.0, 40.0, 44.0, 48.0],
            ],
            [
                [4.0, 6.0, 8.0, 10.0],
                [18.0, 21.0, 24.0, 27.0],
                [40.0, 44.0, 48.0, 52.0],
            ],
        ]);

        let expected2 = TensorData::from([
            [
                [2.0f32, 4.0, 6.0, 8.0],
                [5.0, 6.0, 7.0, 8.0],
                [4.5, 5.0, 5.5, 6.0],
            ],
            [
                [3.0, 6.0, 9.0, 12.0],
                [10.0, 12.0, 14.0, 16.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
        ]);

        result1.to_data().assert_eq(&expected1, true);
        result2.to_data().assert_eq(&expected2, true);
    }
}

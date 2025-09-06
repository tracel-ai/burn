// Include the models for this node type
use crate::include_models;
include_models!(sub, sub_int, sub_shape, sub_broadcast, sub_shape_tensor);

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

    #[test]
    fn sub_broadcast_tensor_ranks() {
        let model: sub_broadcast::Model<TestBackend> = sub_broadcast::Model::default();
        let device = Default::default();

        let x_3d = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [10.0, 20.0, 30.0, 40.0],
                    [50.0, 60.0, 70.0, 80.0],
                    [90.0, 100.0, 110.0, 120.0],
                ],
                [
                    [130.0, 140.0, 150.0, 160.0],
                    [170.0, 180.0, 190.0, 200.0],
                    [210.0, 220.0, 230.0, 240.0],
                ],
            ],
            &device,
        );

        let y_2d = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        );

        let a_2d = Tensor::<TestBackend, 2>::from_floats(
            [
                [100.0, 100.0, 100.0, 100.0],
                [200.0, 200.0, 200.0, 200.0],
                [300.0, 300.0, 300.0, 300.0],
            ],
            &device,
        );

        let b_3d = x_3d.clone();

        let (result1, result2) = model.forward(x_3d, y_2d, a_2d, b_3d);

        let expected1 = TensorData::from([
            [
                [9.0f32, 18.0, 27.0, 36.0],
                [45.0, 54.0, 63.0, 72.0],
                [81.0, 90.0, 99.0, 108.0],
            ],
            [
                [129.0, 138.0, 147.0, 156.0],
                [165.0, 174.0, 183.0, 192.0],
                [201.0, 210.0, 219.0, 228.0],
            ],
        ]);

        let expected2 = TensorData::from([
            [
                [90.0f32, 80.0, 70.0, 60.0],
                [150.0, 140.0, 130.0, 120.0],
                [210.0, 200.0, 190.0, 180.0],
            ],
            [
                [-30.0, -40.0, -50.0, -60.0],
                [30.0, 20.0, 10.0, 0.0],
                [90.0, 80.0, 70.0, 60.0],
            ],
        ]);

        result1.to_data().assert_eq(&expected1, true);
        result2.to_data().assert_eq(&expected2, true);
    }
}

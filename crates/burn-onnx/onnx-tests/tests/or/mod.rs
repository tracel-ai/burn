// Import the shared macro
use crate::include_models;
include_models!(or, or_scalar, or_broadcast);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn or() {
        let device = Default::default();
        let model: or::Model<TestBackend> = or::Model::new(&device);

        let input_x = Tensor::<TestBackend, 4, Bool>::from_bool(
            TensorData::from([[[[false, false, true, true]]]]),
            &device,
        );
        let input_y = Tensor::<TestBackend, 4, Bool>::from_bool(
            TensorData::from([[[[false, true, false, true]]]]),
            &device,
        );

        let output = model.forward(input_x, input_y).to_data();
        let expected = TensorData::from([[[[false, true, true, true]]]]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn or_scalar() {
        let device = Default::default();
        let model: or_scalar::Model<TestBackend> = or_scalar::Model::new(&device);

        // Test various combinations of scalar boolean inputs
        // (input1 || true) || (input2 || false) = true || input2 = true
        assert_eq!(model.forward(false, false), true);
        assert_eq!(model.forward(false, true), true);
        assert_eq!(model.forward(true, false), true);
        assert_eq!(model.forward(true, true), true);
    }

    #[test]
    fn or_broadcast_tensor_ranks() {
        let model = or_broadcast::Model::<TestBackend>::default();
        let device = Default::default();

        // Create tensors matching the Python script
        let x_3d = Tensor::<TestBackend, 3, Bool>::from_data(
            [
                [
                    [true, false, false, false],
                    [true, false, true, false],
                    [false, true, true, true],
                ],
                [
                    [true, false, false, true],
                    [false, false, true, true],
                    [true, true, false, true],
                ],
            ],
            &device,
        );

        let y_2d = Tensor::<TestBackend, 2, Bool>::from_data(
            [
                [false, false, true, true],
                [false, true, true, false],
                [false, false, false, true],
            ],
            &device,
        );

        let a_2d = Tensor::<TestBackend, 2, Bool>::from_data(
            [
                [false, true, false, false],
                [false, false, false, true],
                [true, false, false, true],
            ],
            &device,
        );

        let b_3d = Tensor::<TestBackend, 3, Bool>::from_data(
            [
                [
                    [true, false, true, true],
                    [true, true, false, true],
                    [false, false, false, false],
                ],
                [
                    [true, true, true, false],
                    [false, true, false, false],
                    [false, false, true, false],
                ],
            ],
            &device,
        );

        let (result1, result2) = model.forward(x_3d, y_2d, a_2d, b_3d);

        // Expected outputs from the Python script
        let expected1 = TensorData::from([
            [
                [true, false, true, true],
                [true, true, true, false],
                [false, true, true, true],
            ],
            [
                [true, false, true, true],
                [false, true, true, true],
                [true, true, false, true],
            ],
        ]);
        let expected2 = TensorData::from([
            [
                [true, true, true, true],
                [true, true, false, true],
                [true, false, false, true],
            ],
            [
                [true, true, true, false],
                [false, true, false, true],
                [true, false, true, true],
            ],
        ]);

        result1.to_data().assert_eq(&expected1, true);
        result2.to_data().assert_eq(&expected2, true);
    }
}

// Import the shared macro
use crate::include_models;
include_models!(less, less_scalar, less_broadcast);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn less() {
        let device = Default::default();
        let model: less::Model<TestBackend> = less::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, true, false, false]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn less_scalar() {
        let device = Default::default();
        let model: less_scalar::Model<TestBackend> = less_scalar::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[false, false, false, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn less_broadcast() {
        let device = Default::default();
        let model: less_broadcast::Model<TestBackend> = less_broadcast::Model::new(&device);

        // Shape [1, 77] vs [77, 1] - testing the CLIP-like pattern
        let input1 = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 1.0, -1.0, 2.0, -2.0]], // Using just 5 values for simplicity
            &device,
        );
        let input2 = Tensor::<TestBackend, 2>::from_floats(
            [[0.5], [1.5], [-0.5], [-1.5], [2.5]], // 5x1 shape
            &device,
        );

        let output = model.forward(input1, input2);
        // Expected output shape: [5, 5]
        let expected = TensorData::from([
            [true, false, true, false, true],
            [true, true, true, false, true],
            [false, false, true, false, true],
            [false, false, false, false, true],
            [true, true, true, true, true],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}

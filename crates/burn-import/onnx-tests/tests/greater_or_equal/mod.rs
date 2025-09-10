// Import the shared macro
use crate::include_models;
include_models!(
    greater_or_equal,
    greater_or_equal_scalar,
    greater_or_equal_broadcast
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn greater_or_equal() {
        let device = Default::default();
        let model: greater_or_equal::Model<TestBackend> = greater_or_equal::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, false, true, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn greater_or_equal_scalar() {
        let device = Default::default();
        let model: greater_or_equal_scalar::Model<TestBackend> =
            greater_or_equal_scalar::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[true, true, true, false]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn greater_or_equal_broadcast() {
        let device = Default::default();
        let model: greater_or_equal_broadcast::Model<TestBackend> =
            greater_or_equal_broadcast::Model::new(&device);

        // Shape [4, 1] vs [1, 4]
        let input1 = Tensor::<TestBackend, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);

        let output = model.forward(input1, input2);
        let expected = TensorData::from([
            [true, false, false, false],
            [true, true, false, false],
            [true, true, true, false],
            [true, true, true, true],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}

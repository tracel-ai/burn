// Include the models for this node type
use crate::include_models;
include_models!(
    bitwise_or,
    bitwise_or_scalar,
    scalar_bitwise_or,
    scalar_bitwise_or_scalar
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn bitwise_or_tensors() {
        // Initialize the model
        let device = Default::default();
        let model: bitwise_or::Model<TestBackend> = bitwise_or::Model::new(&device);
        // Run the model
        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 1, 2, 2]], &device);
        let output = model.forward(input1, input2);
        let expected = TensorData::from([[1i64, 3, 3, 6]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitwise_or_scalar_tensor() {
        // Initialize the model
        let device = Default::default();
        let model: bitwise_or_scalar::Model<TestBackend> = bitwise_or_scalar::Model::new(&device);
        // Run the model
        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let scalar = 2;
        let output = model.forward(input1, scalar);
        let expected = TensorData::from([[3i64, 2, 3, 6]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn scalar_bitwise_or_tensor() {
        // Initialize the model
        let device = Default::default();
        let model: scalar_bitwise_or::Model<TestBackend> = scalar_bitwise_or::Model::new(&device);
        // Run the model
        let scalar = 2;
        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(scalar, input2);
        // Bitwise OR is commutative, so result should be same as tensor-scalar
        let expected = TensorData::from([[3i64, 2, 3, 6]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn scalar_bitwise_or_scalar() {
        let device = Default::default();
        let model: scalar_bitwise_or_scalar::Model<TestBackend> =
            scalar_bitwise_or_scalar::Model::new(&device);
        // Run the model
        let lhs = 5; // 0b101
        let rhs = 3; // 0b011
        let output = model.forward(lhs, rhs);
        // 5 | 3 = 7 (0b111)
        assert_eq!(output, 7);
    }
}

// Include the models for this node type
use crate::include_models;
include_models!(
    bitwise_xor,
    bitwise_xor_scalar,
    scalar_bitwise_xor,
    scalar_bitwise_xor_scalar
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn bitwise_xor_tensors() {
        let device = Default::default();
        let model: bitwise_xor::Model<TestBackend> = bitwise_xor::Model::new(&device);
        // Run the model
        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 1, 2, 2]], &device);
        let output = model.forward(input1, input2);
        let expected = TensorData::from([[0i64, 3, 1, 6]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitwise_xor_scalar_tensor() {
        let device = Default::default();
        let model: bitwise_xor_scalar::Model<TestBackend> = bitwise_xor_scalar::Model::new(&device);
        // Run the model
        let input1 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let scalar = 2;
        let output = model.forward(input1, scalar);
        let expected = TensorData::from([[3i64, 0, 1, 6]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn scalar_bitwise_xor_tensor() {
        let device = Default::default();
        let model: scalar_bitwise_xor::Model<TestBackend> = scalar_bitwise_xor::Model::new(&device);
        // Run the model
        let scalar = 2;
        let input2 = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let output = model.forward(scalar, input2);
        // Bitwise XOR is commutative, so result should be same as tensor-scalar
        let expected = TensorData::from([[3i64, 0, 1, 6]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn scalar_bitwise_xor_scalar() {
        let device = Default::default();
        let model: scalar_bitwise_xor_scalar::Model<TestBackend> =
            scalar_bitwise_xor_scalar::Model::new(&device);
        // Run the model
        let lhs = 5; // 0b101
        let rhs = 3; // 0b011
        let output = model.forward(lhs, rhs);
        // 5 ^ 3 = 6 (0b110)
        assert_eq!(output, 6);
    }
}

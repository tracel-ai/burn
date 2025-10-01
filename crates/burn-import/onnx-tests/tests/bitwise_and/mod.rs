// Include the models for this node type
use crate::include_models;
include_models!(
    bitwise_and,
    bitwise_and_scalar,
    scalar_bitwise_and,
    scalar_bitwise_and_scalar
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn bitwise_and_tensors() {
        let device = Default::default();
        let model: bitwise_and::Model<TestBackend> = bitwise_and::Model::new(&device);
        // Run the model
        let input1 = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let input2 = Tensor::<TestBackend, 1, Int>::from_ints([1, 1, 2, 2], &device);
        let output = model.forward(input1, input2);
        let expected = TensorData::from([1i64, 0, 2, 0]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitwise_and_scalar_tensor() {
        let device = Default::default();
        let model: bitwise_and_scalar::Model<TestBackend> = bitwise_and_scalar::Model::new(&device);
        // Run the model
        let input1 = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let scalar = 2;
        let output = model.forward(input1, scalar);
        let expected = TensorData::from([0i64, 2, 2, 0]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn scalar_bitwise_and_tensor() {
        let device = Default::default();
        let model: scalar_bitwise_and::Model<TestBackend> = scalar_bitwise_and::Model::new(&device);
        // Run the model
        let scalar = 2;
        let input2 = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let output = model.forward(scalar, input2);
        // Bitwise AND is commutative, so result should be same as tensor-scalar
        let expected = TensorData::from([0i64, 2, 2, 0]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn scalar_bitwise_and_scalar() {
        let device = Default::default();
        let model: scalar_bitwise_and_scalar::Model<TestBackend> =
            scalar_bitwise_and_scalar::Model::new(&device);
        // Run the model
        let lhs = 7; // 0b111
        let rhs = 3; // 0b011
        let output = model.forward(lhs, rhs);
        // 7 & 3 = 3
        assert_eq!(output, 3);
    }
}

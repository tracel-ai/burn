// Include the models for this node type
use crate::include_models;
include_models!(
    bitshift_left,
    bitshift_left_scalar,
    scalar_bitshift_left,
    scalar_bitshift_left_scalar,
    bitshift_right,
    bitshift_right_scalar,
    scalar_bitshift_right,
    scalar_bitshift_right_scalar
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn bitshift_left_tensors() {
        // Initialize the model with weights (loaded from the exported file)
        let device = Default::default();
        let model: bitshift_left::Model<TestBackend> = bitshift_left::Model::new(&device);
        // Run the model
        let input1 = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let input2 = Tensor::<TestBackend, 1, Int>::from_ints([1, 1, 2, 2], &device);
        let output = model.forward(input1, input2);
        let expected = TensorData::from([2i64, 4, 12, 16]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitshift_left_scalar_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let device = Default::default();
        let model: bitshift_left_scalar::Model<TestBackend> =
            bitshift_left_scalar::Model::new(&device);
        // Run the model
        let input1 = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let scalar = 2;
        let output = model.forward(input1, scalar);
        let expected = TensorData::from([4i64, 8, 12, 16]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitshift_right_tensors() {
        let device = Default::default();
        let model: bitshift_right::Model<TestBackend> = bitshift_right::Model::new(&device);

        // Run the model
        let input1 = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let input2 = Tensor::<TestBackend, 1, Int>::from_ints([1, 1, 2, 2], &device);
        let output = model.forward(input1, input2);
        let expected = TensorData::from([0i64, 1, 0, 1]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn bitshift_right_scalar_tensor() {
        // Initialize the model with weights (loaded from the exported file)
        let device = Default::default();
        let model: bitshift_right_scalar::Model<TestBackend> =
            bitshift_right_scalar::Model::new(&device);
        // Run the model
        let input1 = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let scalar = 2;
        let output = model.forward(input1, scalar);
        let expected = TensorData::from([0i64, 0, 0, 1]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn scalar_bitshift_left_tensor() {
        let device = Default::default();
        let model: scalar_bitshift_left::Model<TestBackend> =
            scalar_bitshift_left::Model::new(&device);
        // Run the model
        let scalar = 4;
        let shift_amounts = Tensor::<TestBackend, 1, Int>::from_ints([1, 1, 2, 2], &device);
        let output = model.forward(scalar, shift_amounts);
        // 4 << 1 = 8, 4 << 1 = 8, 4 << 2 = 16, 4 << 2 = 16
        let expected = TensorData::from([8i64, 8, 16, 16]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn scalar_bitshift_right_tensor() {
        let device = Default::default();
        let model: scalar_bitshift_right::Model<TestBackend> =
            scalar_bitshift_right::Model::new(&device);
        // Run the model
        let scalar = 8;
        let shift_amounts = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let output = model.forward(scalar, shift_amounts);
        // 8 >> 1 = 4, 8 >> 2 = 2, 8 >> 3 = 1, 8 >> 4 = 0
        let expected = TensorData::from([4i64, 2, 1, 0]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn scalar_bitshift_left_scalar() {
        let device = Default::default();
        let model: scalar_bitshift_left_scalar::Model<TestBackend> =
            scalar_bitshift_left_scalar::Model::new(&device);
        // Run the model
        let lhs = 4;
        let rhs = 2;
        let output = model.forward(lhs, rhs);
        // 4 << 2 = 16
        assert_eq!(output, 16);
    }

    #[test]
    fn scalar_bitshift_right_scalar() {
        let device = Default::default();
        let model: scalar_bitshift_right_scalar::Model<TestBackend> =
            scalar_bitshift_right_scalar::Model::new(&device);
        // Run the model
        let lhs = 16;
        let rhs = 2;
        let output = model.forward(lhs, rhs);
        // 16 >> 2 = 4
        assert_eq!(output, 4);
    }
}

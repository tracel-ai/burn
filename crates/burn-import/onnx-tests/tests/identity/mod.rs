// Include the models for this node type
use crate::include_models;
include_models!(
    identity_constant,
    identity_passthrough,
    identity_chain,
    identity_only
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_identity_constant() {
        let model: identity_constant::Model<TestBackend> = identity_constant::Model::default();

        let output = model.forward();

        // Verify output shape is correct (3-element vector)
        assert_eq!(output.shape().dims, [3]);

        // Verify the constant values [1.0, 2.0, 3.0] are preserved
        let expected = TensorData::from([1.0f32, 2.0, 3.0]);
        output.to_data().assert_eq(&expected, true);

        // The test verifies that Identity node with constant input
        // was properly converted to a Constant node
    }

    #[test]
    fn test_identity_passthrough() {
        let model: identity_passthrough::Model<TestBackend> =
            identity_passthrough::Model::default();

        let device = Default::default();

        // Create input tensor (2x3 as defined in the Python script)
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input);

        // Verify output shape is correct (2x3)
        assert_eq!(output.shape().dims, [2, 3]);

        // Verify the Identity passthrough was removed and Add operation works
        // Input [[1,2,3],[4,5,6]] + constant [[1,1,1],[1,1,1]] = [[2,3,4],[5,6,7]]
        let expected = TensorData::from([[2.0f32, 3.0, 4.0], [5.0, 6.0, 7.0]]);
        output.to_data().assert_eq(&expected, true);

        // The test verifies that Identity node without constant input
        // was properly removed and inputs remapped correctly
    }

    #[test]
    fn test_identity_chain() {
        let model: identity_chain::Model<TestBackend> = identity_chain::Model::default();

        let device = Default::default();

        // Create input tensor (2x3 as defined in the Python script)
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0f32, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            &device,
        );

        let output = model.forward(input);

        // Verify output shape is correct (2x3)
        assert_eq!(output.shape().dims, [2, 3]);

        // Verify the chain of Identity nodes was removed and ReLU works
        // ReLU of [[1,-2,3],[-4,5,-6]] = [[1,0,3],[0,5,0]]
        let expected = TensorData::from([[1.0f32, 0.0, 3.0], [0.0, 5.0, 0.0]]);
        output.to_data().assert_eq(&expected, true);

        // The test verifies that multiple Identity nodes in sequence
        // were properly removed and the final operation works correctly
    }

    #[test]
    fn test_identity_only() {
        let model: identity_only::Model<TestBackend> = identity_only::Model::default();

        let device = Default::default();

        // Create input tensor (3x4 as defined in the Python script)
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]),
            &device,
        );

        let output = model.forward(input.clone());

        // Verify output shape is correct (3x4)
        assert_eq!(output.shape().dims, [3, 4]);

        // Verify the Identity operation passes through the input unchanged
        // Since Identity should be removed, this should be a direct passthrough
        input.to_data().assert_eq(&output.to_data(), true);

        // The test verifies that a standalone Identity node without constant input
        // is properly removed and results in direct input-to-output mapping
    }
}

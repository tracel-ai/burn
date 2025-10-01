// Include the models for this node type
use crate::include_models;
include_models!(initializer_to_const);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_initializer_to_const() {
        let model: initializer_to_const::Model<TestBackend> =
            initializer_to_const::Model::default();

        let device = Default::default();

        // Create input tensor (2x3 as defined in the Python script)
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0f32, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            &device,
        );

        let output = model.forward(input);

        // Verify output shape is correct (2x3)
        assert_eq!(output.shape().dims, [2, 3]);

        // Verify the addition worked correctly
        // Input [[1,1,1],[1,1,1]] + initializer [[1,2,3],[4,5,6]] = [[2,3,4],[5,6,7]]
        let expected = TensorData::from([[2.0f32, 3.0, 4.0], [5.0, 6.0, 7.0]]);
        output.to_data().assert_eq(&expected, true);

        // The test verifies that the model can execute successfully,
        // which means initializers were properly converted to constants
        // and the convert_initializer_inputs_to_constants function worked correctly
    }
}

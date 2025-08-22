// Include the models for this node type
use crate::include_models;
include_models!(constant_lifting_multiple);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_constant_lifting_multiple() {
        let model: constant_lifting_multiple::Model<TestBackend> =
            constant_lifting_multiple::Model::default();

        let device = Default::default();

        // Create input tensor (2x3 as defined in the Python script)
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0f32, -2.0, 8.0], [-4.0, 5.0, 3.0]]),
            &device,
        );

        let output = model.forward(input);

        // Verify output shape is correct (2x3)
        assert_eq!(output.shape().dims, [2, 3]);

        // Verify the Clip operation works with lifted constants (min=0.0, max=6.0)
        // Clip of [[1,-2,8],[-4,5,3]] with min=0.0, max=6.0 = [[1,0,6],[0,5,3]]
        let expected = TensorData::from([[1.0f32, 0.0, 6.0], [0.0, 5.0, 3.0]]);
        output.to_data().assert_eq(&expected, true);

        // The test verifies that multiple Constant nodes feeding into a Clip node
        // were properly lifted and the constant lifting mechanism works correctly
    }
}

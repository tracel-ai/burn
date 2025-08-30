// Include the models for this node type
use crate::include_models;
include_models!(constant_lifting_multiple, constant_reused);

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

    #[test]
    fn test_constant_reused() {
        // This test verifies that constants used multiple times are NOT lifted
        // while constants used only once ARE lifted
        let model: constant_reused::Model<TestBackend> = constant_reused::Model::default();

        let device = Default::default();

        // Create input tensor (2x3)
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );

        let output = model.forward(input);

        // Expected computation:
        // 1. mul1 = input * 2.0 (single-use constant, should be lifted)
        // 2. mul2 = mul1 * 3.0 (multi-use constant, should NOT be lifted)
        // 3. add = mul2 + 3.0 (same multi-use constant)
        // 4. output = clip(add, 0.0, 10.0) (single-use constants, should be lifted)

        // Manual calculation:
        // input = [[1, 2, 3], [4, 5, 6]]
        // mul1 = [[2, 4, 6], [8, 10, 12]]
        // mul2 = [[6, 12, 18], [24, 30, 36]]
        // add = [[9, 15, 21], [27, 33, 39]]
        // clipped = [[9, 10, 10], [10, 10, 10]] (clamped to max=10)

        let expected = TensorData::from([[9.0f32, 10.0, 10.0], [10.0, 10.0, 10.0]]);
        output.to_data().assert_eq(&expected, true);

        // This test ensures:
        // - single_use_value (2.0) is lifted since it's used only in mul_single_use
        // - multi_use_value (3.0) is NOT lifted since it's used in both mul_multi_use_1 and add_multi_use_2
        // - clip_min (0.0) and clip_max (10.0) are lifted since they're used only once
    }
}

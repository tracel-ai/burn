#[burn_tensor_testgen::testgen(padding)]
mod tests {
    use super::*;
    use burn_tensor::{
        Numeric, Shape, Tensor, TensorData, as_type,
        backend::Backend,
        ops::PadMode,
        tests::{Float as _, Int as _},
    };

    #[test]
    fn padding_constant_2d_test() {
        let unpadded_floats: [[f32; 3]; 2] = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
        let tensor = TestTensor::<2>::from(unpadded_floats);

        let padded_tensor = tensor.pad((2, 2, 2, 2), PadMode::Constant(1.1));

        let expected = TensorData::from(as_type!(FloatType: [
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 0.0, 1.0, 2.0, 1.1, 1.1],
            [1.1, 1.1, 3.0, 4.0, 5.0, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        ]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_constant_4d_test() {
        let unpadded_floats = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]];
        let tensor = TestTensor::<4>::from(unpadded_floats);

        let padded_tensor = tensor.pad((2, 2, 2, 2), PadMode::Constant(1.1));

        let expected = TensorData::from(as_type!(FloatType: [[[
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 0.0, 1.0, 1.1, 1.1],
            [1.1, 1.1, 2.0, 3.0, 1.1, 1.1],
            [1.1, 1.1, 4.0, 5.0, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        ]]]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_constant_asymmetric_test() {
        let unpadded_floats = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]];
        let tensor = TestTensor::<4>::from(unpadded_floats);

        let padded_tensor = tensor.pad((2, 1, 4, 3), PadMode::Constant(1.1));

        let expected = TensorData::from(as_type!(FloatType: [[[
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 0.0, 1.0, 1.1],
            [1.1, 1.1, 2.0, 3.0, 1.1],
            [1.1, 1.1, 4.0, 5.0, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1],
        ]]]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_reflect_2d_test() {
        // Test reflect padding on a 2D tensor
        // Input: [[1, 2, 3], [4, 5, 6]]
        // With padding (1, 1, 1, 1):
        // - Top: reflect row 1 -> [4, 5, 6]
        // - Bottom: reflect row 0 -> [1, 2, 3]
        // - Left: reflect col 1
        // - Right: reflect col 1
        let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::Reflect);

        // Expected: reflect excludes the edge value
        // Before padding height: [[1,2,3], [4,5,6]]
        // After top pad (reflect row at index 1): [[4,5,6], [1,2,3], [4,5,6]]
        // After bottom pad (reflect row at index 1 from end): [[4,5,6], [1,2,3], [4,5,6], [1,2,3]]
        // Then pad width similarly
        let expected = TensorData::from(as_type!(FloatType: [
            [5.0, 4.0, 5.0, 6.0, 5.0],
            [2.0, 1.0, 2.0, 3.0, 2.0],
            [5.0, 4.0, 5.0, 6.0, 5.0],
            [2.0, 1.0, 2.0, 3.0, 2.0],
        ]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_reflect_width_only_test() {
        // Test reflect padding on width dimension only
        let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0]]);

        let padded_tensor = tensor.pad((2, 2, 0, 0), PadMode::Reflect);

        // Input: [1, 2, 3, 4]
        // Reflect left 2: take indices [1, 2] = [2, 3], flip = [3, 2]
        // Reflect right 2: take indices [1, 2] from end = [2, 3], flip = [3, 2]
        // Result: [3, 2, 1, 2, 3, 4, 3, 2]
        let expected =
            TensorData::from(as_type!(FloatType: [[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_reflect_4d_test() {
        // Test reflect padding on 4D tensor (common for images: NCHW)
        let tensor = TestTensor::<4>::from([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]);

        let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::Reflect);

        let expected = TensorData::from(as_type!(FloatType: [[[[
            5.0, 4.0, 5.0, 6.0, 5.0],
            [2.0, 1.0, 2.0, 3.0, 2.0],
            [5.0, 4.0, 5.0, 6.0, 5.0],
            [8.0, 7.0, 8.0, 9.0, 8.0],
            [5.0, 4.0, 5.0, 6.0, 5.0
        ]]]]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_edge_2d_test() {
        // Test edge padding on a 2D tensor
        let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::Edge);

        // Edge padding replicates the boundary values
        let expected = TensorData::from(as_type!(FloatType: [
            [1.0, 1.0, 2.0, 3.0, 3.0],
            [1.0, 1.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 6.0, 6.0],
            [4.0, 4.0, 5.0, 6.0, 6.0],
        ]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_edge_width_only_test() {
        // Test edge padding on width dimension only
        let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0]]);

        let padded_tensor = tensor.pad((2, 3, 0, 0), PadMode::Edge);

        // Input: [1, 2, 3, 4]
        // Left 2: [1, 1]
        // Right 3: [4, 4, 4]
        // Result: [1, 1, 1, 2, 3, 4, 4, 4, 4]
        let expected =
            TensorData::from(as_type!(FloatType: [[1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0]]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_edge_4d_test() {
        // Test edge padding on 4D tensor
        let tensor = TestTensor::<4>::from([[[[1.0, 2.0], [3.0, 4.0]]]]);

        let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::Edge);

        let expected = TensorData::from(as_type!(FloatType: [[[[
            1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0
        ]]]]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn padding_constant_default_test() {
        // Test default PadMode (Constant with 0.0)
        let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

        let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::default());

        let expected = TensorData::from(as_type!(FloatType: [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]));
        padded_tensor.into_data().assert_eq(&expected, false);
    }
}

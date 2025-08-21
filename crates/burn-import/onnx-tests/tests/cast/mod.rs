// Import the shared macro
use crate::include_models;
include_models!(cast, cast_shape, cast_shape_to_float, cast_shape_to_bool);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Int, Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn cast() {
        let device = Default::default();
        let model: cast::Model<TestBackend> = cast::Model::new(&device);

        let input_bool =
            Tensor::<TestBackend, 2, Bool>::from_bool(TensorData::from([[true], [true]]), &device);
        let input_int = Tensor::<TestBackend, 2, Int>::from_ints([[1], [1]], &device);
        let input_float = Tensor::<TestBackend, 2>::from_floats([[1f32], [1.]], &device);
        let input_scalar = 1f32;

        let (
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output_scalar,
        ) = model.forward(
            input_bool.clone(),
            input_int.clone(),
            input_float.clone(),
            input_scalar,
        );
        let expected_bool = input_bool.to_data();
        let expected_int = input_int.to_data();
        let expected_float = input_float.to_data();
        let expected_scalar = 1;

        output1.to_data().assert_eq(&expected_bool, true);
        output2.to_data().assert_eq(&expected_int, true);
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected_float, Tolerance::default());

        output4.to_data().assert_eq(&expected_bool, true);
        output5.to_data().assert_eq(&expected_int, true);
        output6
            .to_data()
            .assert_approx_eq::<FT>(&expected_float, Tolerance::default());

        output7.to_data().assert_eq(&expected_bool, true);
        output8.to_data().assert_eq(&expected_int, true);
        output9
            .to_data()
            .assert_approx_eq::<FT>(&expected_float, Tolerance::default());

        assert_eq!(output_scalar, expected_scalar);
    }

    #[test]
    fn cast_shape() {
        // This test verifies that Shape types maintain their i64 representation
        // even when cast to other integer types in ONNX

        let device = Default::default();
        let model: cast_shape::Model<TestBackend> = cast_shape::Model::new(&device);

        // Create test input
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);

        // Run the model
        let (shape_original, shape_casted) = model.forward(input);

        // Verify both shapes are the same [i64; 3] arrays
        // In Burn, Shape types always use i64 regardless of ONNX cast operations
        assert_eq!(shape_original, [2i64, 3i64, 4i64]);
        assert_eq!(shape_casted, [2i64, 3i64, 4i64]); // Cast to int32 is a no-op for Shape types
    }

    #[test]
    fn cast_shape_to_float() {
        // This test verifies that Shape types can be cast to float tensors
        // When casting Shape to float, it should convert to a 1D tensor

        let device = Default::default();
        let model: cast_shape_to_float::Model<TestBackend> =
            cast_shape_to_float::Model::new(&device);

        // Create test input with shape [2, 3, 4]
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let multiplier = Tensor::<TestBackend, 1>::from_floats([2.0], &device);

        // Run the model - it extracts shape, casts to float, and multiplies by 2
        let output = model.forward(input, multiplier);

        // The output should be [2.0, 3.0, 4.0] * 2.0 = [4.0, 6.0, 8.0]
        let expected = Tensor::<TestBackend, 1>::from_floats([4.0, 6.0, 8.0], &device);

        // Check values are close
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn cast_shape_to_bool() {
        // This test verifies that Shape types can be cast to bool tensors
        // When casting Shape to bool, non-zero values become true

        let device = Default::default();
        let model: cast_shape_to_bool::Model<TestBackend> = cast_shape_to_bool::Model::new(&device);

        // Create test input with shape [2, 3, 4] (all non-zero dimensions)
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);

        // Run the model - it extracts shape and casts to bool
        let output = model.forward(input);

        // The output should be [true, true, true] since all dimensions are non-zero
        let expected = Tensor::<TestBackend, 1, Bool>::from_bool(
            TensorData::from([true, true, true]),
            &device,
        );

        // Check values match
        output.to_data().assert_eq(&expected.to_data(), true);
    }
}

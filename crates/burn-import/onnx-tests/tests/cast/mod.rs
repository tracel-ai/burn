// Import the shared macro
use crate::include_models;
include_models!(cast, cast_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Int, Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::Backend;
    type FT = FloatElem<Backend>;

    #[test]
    fn cast() {
        let device = Default::default();
        let model: cast::Model<Backend> = cast::Model::new(&device);

        let input_bool =
            Tensor::<Backend, 2, Bool>::from_bool(TensorData::from([[true], [true]]), &device);
        let input_int = Tensor::<Backend, 2, Int>::from_ints([[1], [1]], &device);
        let input_float = Tensor::<Backend, 2>::from_floats([[1f32], [1.]], &device);
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
        let model: cast_shape::Model<Backend> = cast_shape::Model::new(&device);

        // Create test input
        let input = Tensor::<Backend, 3>::ones([2, 3, 4], &device);

        // Run the model
        let (shape_original, shape_casted) = model.forward(input);

        // Verify both shapes are the same [i64; 3] arrays
        // In Burn, Shape types always use i64 regardless of ONNX cast operations
        assert_eq!(shape_original, [2i64, 3i64, 4i64]);
        assert_eq!(shape_casted, [2i64, 3i64, 4i64]); // Cast to int32 is a no-op for Shape types
    }
}

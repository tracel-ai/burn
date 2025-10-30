// Import the shared macro
use crate::include_models;
include_models!(
    constant_of_shape,
    constant_of_shape_full_like,
    constant_of_shape_scalar,
    constant_of_shape_scalar_custom_value,
    constant_of_shape_tensor,
    constant_of_shape_shape_optimization,
    constant_of_shape_with_constant_input
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn constant_of_shape() {
        // This tests shape is being passed directly to the model
        let device = Default::default();
        let model = constant_of_shape::Model::<TestBackend>::new(&device);
        let input_shape = [2i64, 3i64, 2i64];
        let expected = Tensor::<TestBackend, 3>::full([2, 3, 2], 1.125, &device).to_data();

        let output = model.forward(input_shape);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn constant_of_shape_full_like() {
        // This tests shape is being passed from the input tensor
        let device = Default::default();
        let model = constant_of_shape_full_like::Model::<TestBackend>::new(&device);
        let shape = [2, 3, 2];
        let f_expected = Tensor::<TestBackend, 3>::full(shape, 3.0, &device);
        let i_expected = Tensor::<TestBackend, 3, Int>::full(shape, 5, &device);
        let b_expected = Tensor::<TestBackend, 3, Int>::ones(shape, &device).bool();

        let input = Tensor::ones(shape, &device);
        let (f_output, i_output, b_output) = model.forward(input);

        f_output.to_data().assert_eq(&f_expected.to_data(), true);
        i_output.to_data().assert_eq(&i_expected.to_data(), true);
        b_output.to_data().assert_eq(&b_expected.to_data(), true);
    }

    #[test]
    fn constant_of_shape_scalar_test() {
        // Test scalar output case
        let device = Default::default();
        let model = constant_of_shape_scalar::Model::<TestBackend>::new(&device);

        // No runtime inputs - shape comes from initializer
        let output: f32 = model.forward();

        // Output should be a scalar with value 0.0 (default)
        assert_eq!(output, 0.0f32);
    }

    #[test]
    fn constant_of_shape_scalar_custom_value_test() {
        // Test scalar output with custom value
        let device = Default::default();
        let model = constant_of_shape_scalar_custom_value::Model::<TestBackend>::new(&device);

        // No runtime inputs - shape comes from initializer
        let output: i64 = model.forward();

        // Output should be a scalar with value 42
        assert_eq!(output, 42i64);
    }

    #[test]
    fn constant_of_shape_tensor_test() {
        // Test tensor output case
        let device = Default::default();
        let model = constant_of_shape_tensor::Model::<TestBackend>::new(&device);

        // Input is shape [2, 3]
        let shape_input = [2i64, 3i64];
        let output = model.forward(shape_input);

        // Output should be a 2x3 tensor filled with 0.0 (default)
        assert_eq!(output.dims(), [2, 3]);
        let expected = Tensor::<TestBackend, 2>::zeros([2, 3], &device);
        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn constant_of_shape_shape_optimization_test() {
        // Test Shape(1) -> Shape(1) optimization with Int64
        let device = Default::default();
        let model = constant_of_shape_shape_optimization::Model::<TestBackend>::new(&device);

        // No runtime inputs - shape [3] comes from initializer
        let output: [i64; 1] = model.forward();

        // Output should be [5] (one element with value 5, optimized from Shape(1))
        assert_eq!(output, [5i64]);
    }

    #[test]
    fn constant_of_shape_with_constant_input_test() {
        // Test ConstantOfShape where the shape comes from a Constant node
        // This tests the constant lifting mechanism where the shape values
        // are known at compile time and embedded directly in the generated code
        let device = Default::default();
        let model = constant_of_shape_with_constant_input::Model::<TestBackend>::new(&device);

        // Model has no inputs - the shape [2, 3, 4] comes from a constant
        let output = model.forward();

        // Output should be a 2x3x4 tensor filled with 1 (as specified in the value attribute)
        assert_eq!(output.dims(), [2, 3, 4]);
        let expected = Tensor::<TestBackend, 3, Int>::full([2, 3, 4], 1i64, &device);
        output.to_data().assert_eq(&expected.to_data(), true);
    }
}

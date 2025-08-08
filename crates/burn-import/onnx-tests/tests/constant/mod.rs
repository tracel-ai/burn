// Import the shared macro
use crate::include_models;
include_models!(
    constant_f32,
    constant_f64,
    constant_i32,
    constant_i64,
    constant_shape
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Shape, Tensor};

    use crate::backend::Backend;

    #[test]
    fn add_constant_f32() {
        let device = Default::default();
        let model = constant_f32::Model::<Backend>::new(&device);
        let input = Tensor::<Backend, 3>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<Backend, 3>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_constant_f64() {
        let device = Default::default();
        let model = constant_f64::Model::<Backend>::new(&device);
        let input = Tensor::<Backend, 3>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<Backend, 3>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_constant_i32() {
        let device = Default::default();
        let model = constant_i32::Model::<Backend>::new(&device);
        let input = Tensor::<Backend, 3, Int>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<Backend, 3, Int>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn add_constant_i64() {
        let device = Default::default();
        let model = constant_i64::Model::<Backend>::new(&device);
        let input = Tensor::<Backend, 3, Int>::zeros(Shape::from([2, 3, 4]), &device);
        let expected = Tensor::<Backend, 3, Int>::full([2, 3, 4], 2, &device).to_data();

        let output = model.forward(input);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn constant_shape() {
        let device = Default::default();
        let model = constant_shape::Model::<Backend>::new(&device);

        // Create input tensor with shape [2, 4, 6]
        let input = Tensor::<Backend, 3>::zeros(Shape::from([2, 4, 6]), &device);

        // The model tests Shape operations with constants
        // Input shape: [2, 4, 6]
        // Scalar constant: 2
        // Shape constant: [1, 2, 3]
        let (shape_add_scalar, shape_mul_scalar, shape_add_shape, shape_mul_shape) =
            model.forward(input);

        // Check shape_add_scalar: [2, 4, 6] + 2 = [4, 6, 8]
        assert_eq!(shape_add_scalar[0], 4);
        assert_eq!(shape_add_scalar[1], 6);
        assert_eq!(shape_add_scalar[2], 8);

        // Check shape_mul_scalar: [2, 4, 6] * 2 = [4, 8, 12]
        assert_eq!(shape_mul_scalar[0], 4);
        assert_eq!(shape_mul_scalar[1], 8);
        assert_eq!(shape_mul_scalar[2], 12);

        // Check shape_add_shape: [2, 4, 6] + [1, 2, 3] = [3, 6, 9]
        assert_eq!(shape_add_shape[0], 3);
        assert_eq!(shape_add_shape[1], 6);
        assert_eq!(shape_add_shape[2], 9);

        // Check shape_mul_shape: [2, 4, 6] * [1, 2, 3] = [2, 8, 18]
        assert_eq!(shape_mul_shape[0], 2);
        assert_eq!(shape_mul_shape[1], 8);
        assert_eq!(shape_mul_shape[2], 18);
    }
}

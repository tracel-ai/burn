use crate::include_models;
include_models!(
    squeeze,
    squeeze_multiple,
    squeeze_shape,
    squeeze_shape_noop,
    squeeze_scalar,
    squeeze_float,
    squeeze_tensor_to_scalar,
    squeeze_opset13_axes_input,
    squeeze_no_axes
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};

    use crate::backend::TestBackend;

    #[test]
    fn squeeze() {
        let device = Default::default();
        let model = squeeze::Model::<TestBackend>::new(&device);
        let input_shape = Shape::from([3, 4, 1, 5]);
        let expected_shape = Shape::from([3, 4, 5]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn squeeze_multiple() {
        let device = Default::default();
        let model = squeeze_multiple::Model::<TestBackend>::new(&device);
        let input_shape = Shape::from([3, 4, 1, 5, 1]);
        let expected_shape = Shape::from([3, 4, 5]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn squeeze_shape() {
        let device = Default::default();
        let model = squeeze_shape::Model::<TestBackend>::new(&device);
        // Input tensor is 3x4x5
        let input = Tensor::<TestBackend, 3>::ones([3, 4, 5], &device);
        // The model: Shape -> Slice(0:1) -> Squeeze
        // Expected: [3, 4, 5] -> [3] -> 3
        let output = model.forward(input);
        assert_eq!(output, 3i64);
    }

    #[test]
    fn squeeze_shape_noop() {
        let device = Default::default();
        let model = squeeze_shape_noop::Model::<TestBackend>::new(&device);
        // Input tensor is 6x7
        let input = Tensor::<TestBackend, 2>::ones([6, 7], &device);
        // The model: Shape -> Squeeze(axis=0)
        // Expected: [6, 7] -> [6, 7] (no-op since axis 0 has size 6, not 1)
        let output = model.forward(input);
        assert_eq!(output, [6, 7]);
    }

    #[test]
    fn squeeze_scalar() {
        let device = Default::default();
        let model = squeeze_scalar::Model::<TestBackend>::new(&device);
        // The model has a constant scalar 1.5 that gets squeezed
        // Expected: 1.5 -> 1.5 (no-op)
        let output = model.forward();
        assert_eq!(output, 1.5f32);
    }

    #[test]
    fn squeeze_float() {
        // Test verifies that the improved squeeze implementation using .into_scalar()
        // works correctly for float tensors with .elem::<f32>() casting
        let device = Default::default();
        let model = squeeze_float::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 1>::from_data([14159.222f32], &device);
        let output = model.forward(input);
        assert!((output - 14159.222f32).abs() < 1e-6);
    }

    #[test]
    fn squeeze_tensor_to_scalar() {
        // Test squeezing a multi-dimensional tensor [1, 1, 1] with one element to a scalar
        let device = Default::default();
        let model = squeeze_tensor_to_scalar::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::from_data([[[42.5f32]]], &device);
        let output = model.forward(input);
        assert!((output - 42.5f32).abs() < 1e-6);
    }

    #[test]
    fn squeeze_opset13_axes_input() {
        // Test ONNX opset 13+ style where axes are provided as input instead of attribute
        // This simulates the FaceNet512 model case with GlobalAvgPool output
        let device = Default::default();
        let model = squeeze_opset13_axes_input::Model::<TestBackend>::new(&device);
        let input_shape = Shape::from([1, 512, 1, 1]);
        let expected_shape = Shape::from([1, 512]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn squeeze_no_axes() {
        // Test squeeze without axes specified - should squeeze all dimensions with size 1
        // Input shape: [2, 1, 3, 1, 4]
        // Output shape: [2, 3, 4]
        let device = Default::default();
        let model = squeeze_no_axes::Model::<TestBackend>::new(&device);
        let input_shape = Shape::from([2, 1, 3, 1, 4]);
        let expected_shape = Shape::from([2, 3, 4]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }
}

use crate::include_models;
include_models!(squeeze, squeeze_multiple, squeeze_shape, squeeze_shape_noop);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};

    use crate::backend::Backend;

    #[test]
    fn squeeze() {
        let device = Default::default();
        let model = squeeze::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 1, 5]);
        let expected_shape = Shape::from([3, 4, 5]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn squeeze_multiple() {
        let device = Default::default();
        let model = squeeze_multiple::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 1, 5, 1]);
        let expected_shape = Shape::from([3, 4, 5]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn squeeze_shape() {
        let device = Default::default();
        let model = squeeze_shape::Model::<Backend>::new(&device);
        // Input tensor is 3x4x5
        let input = Tensor::<Backend, 3>::ones([3, 4, 5], &device);
        // The model: Shape -> Slice(0:1) -> Squeeze
        // Expected: [3, 4, 5] -> [3] -> 3
        let output = model.forward(input);
        assert_eq!(output, 3i64);
    }

    #[test]
    fn squeeze_shape_noop() {
        let device = Default::default();
        let model = squeeze_shape_noop::Model::<Backend>::new(&device);
        // Input tensor is 6x7x8x9
        let input = Tensor::<Backend, 4>::ones([6, 7, 8, 9], &device);
        // The model: Shape -> Slice(0:2) -> Squeeze(axis=0)
        // Expected: [6, 7, 8, 9] -> [6, 7] -> [6, 7] (no-op)
        let output = model.forward(input);
        assert_eq!(output, [6, 7]);
    }
}

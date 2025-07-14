use crate::include_models;
include_models!(unsqueeze_like, unsqueeze_runtime_axes);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};

    use crate::backend::Backend;

    #[test]
    fn unsqueeze_runtime_axes() {
        let device = Default::default();
        let model: unsqueeze_runtime_axes::Model<Backend> =
            unsqueeze_runtime_axes::Model::new(&device);
        let input_shape = Shape::from([3, 4, 5]);
        let expected_shape = Shape::from([1, 3, 1, 4, 5, 1]);
        let input = Tensor::ones(input_shape, &device);

        // Note: The axes tensor must have rank 1 with a single element
        // as the generated ONNX requires a 1D tensor for static shape operations
        // see unsqueeze.onnx
        let axes = Tensor::from_ints([2], &device);
        let output = model.forward(input, axes);
        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn unsqueeze_like() {
        let device = Default::default();
        let model = unsqueeze_like::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 5]);
        let expected_shape = Shape::from([3, 4, 5, 1]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input, 1.0);
        assert_eq!(expected_shape, output.0.shape());
        assert_eq!(Shape::from([1]), output.1.shape());
    }
}

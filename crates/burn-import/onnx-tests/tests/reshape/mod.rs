use crate::include_models;
include_models!(reshape, reshape_with_1d_tensor, reshape_with_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn reshape() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: reshape::Model<Backend> = reshape::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 1>::from_floats([0., 1., 2., 3.], &device);
        let output = model.forward(input);
        let expected = TensorData::from([[0f32, 1., 2., 3.]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn reshape_with_1d_tensor() {
        // This test verifies that dynamic reshape operations correctly infer
        // the output shape from the ONNX model specification.

        // Initialize the model
        let device = Default::default();
        let model: reshape_with_1d_tensor::Model<Backend> =
            reshape_with_1d_tensor::Model::new(&device);

        // Run the model with shape as tensor input
        let input = Tensor::<Backend, 1>::from_floats(
            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            &device,
        );
        let shape = Tensor::<Backend, 1, burn::tensor::Int>::from_ints([3, 4], &device);
        let output = model.forward(input, shape);

        // Output should be 2D with shape [3, 4] as specified in the ONNX model
        let expected = TensorData::from([[0f32, 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn reshape_with_shape() {
        // This test verifies that reshape can accept a Shape type (from Shape node)
        // as its second input in addition to a 1D tensor.

        // Initialize the model
        let device = Default::default();
        let model: reshape_with_shape::Model<Backend> = reshape_with_shape::Model::new(&device);

        // Run the model with input and shape_source tensors
        let input = Tensor::<Backend, 1>::from_floats(
            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            &device,
        );
        // shape_source is used to extract shape via Shape node
        let shape_source = Tensor::<Backend, 2>::zeros([3, 4], &device);
        let output = model.forward(input, shape_source);

        // Output should be 2D with shape [3, 4] extracted from shape_source
        let expected = TensorData::from([[0f32, 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]]);
        output.to_data().assert_eq(&expected, true);
    }
}

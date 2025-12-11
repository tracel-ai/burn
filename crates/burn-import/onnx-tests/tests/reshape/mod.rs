use crate::include_models;
include_models!(
    reshape,
    reshape_with_1d_tensor,
    reshape_with_shape,
    reshape_to_scalar,
    reshape_3d_to_scalar,
    reshape_shape_to_shape,
    reshape_shape_with_neg,
    reshape_shape_partial,
    reshape_scalar_to_scalar
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn reshape() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: reshape::Model<TestBackend> = reshape::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 1>::from_floats([0., 1., 2., 3.], &device);
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
        let model: reshape_with_1d_tensor::Model<TestBackend> =
            reshape_with_1d_tensor::Model::new(&device);

        // Run the model with shape as tensor input
        let input = Tensor::<TestBackend, 1>::from_floats(
            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            &device,
        );
        let shape = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints([3, 4], &device);
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
        let model: reshape_with_shape::Model<TestBackend> = reshape_with_shape::Model::new(&device);

        // Run the model with input and shape_source tensors
        let input = Tensor::<TestBackend, 1>::from_floats(
            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            &device,
        );
        // shape_source is used to extract shape via Shape node
        let shape_source = Tensor::<TestBackend, 2>::zeros([3, 4], &device);
        let output = model.forward(input, shape_source);

        // Output should be 2D with shape [3, 4] extracted from shape_source
        let expected = TensorData::from([[0f32, 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn reshape_to_scalar_test() {
        // This test verifies that reshape can convert a 1x1 tensor to a scalar

        // Initialize the model
        let device = Default::default();
        let model: reshape_to_scalar::Model<TestBackend> = reshape_to_scalar::Model::new(&device);

        // Run the model with a 1x1 tensor input
        let input = Tensor::<TestBackend, 2>::from_floats([[1.5]], &device);
        let output = model.forward(input);

        // Output should be a scalar value
        assert_eq!(output, 1.5f32);
    }

    #[test]
    fn reshape_3d_to_scalar_test() {
        // This test verifies that reshape can convert a 1x1x1 tensor to a scalar

        // Initialize the model
        let device = Default::default();
        let model: reshape_3d_to_scalar::Model<TestBackend> =
            reshape_3d_to_scalar::Model::new(&device);

        // Run the model with a 1x1x1 tensor input
        let input = Tensor::<TestBackend, 3>::from_floats([[[2.5]]], &device);
        let output = model.forward(input);

        // Output should be a scalar value
        assert_eq!(output, 2.5f32);
    }

    #[test]
    fn reshape_shape_to_shape_test() {
        // This test verifies that reshape can accept a Shape type directly as input
        // and output a Shape type (Shape -> Shape path)

        // Initialize the model
        let device = Default::default();
        let model: reshape_shape_to_shape::Model<TestBackend> =
            reshape_shape_to_shape::Model::new(&device);

        // Run the model with a tensor whose shape will be extracted and reshaped
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 4], &device);
        let output = model.forward(input);

        // Output should be [2, 3, 4] - the shape of the input tensor
        assert_eq!(output, [2i64, 3, 4]);
    }

    #[test]
    fn reshape_shape_with_neg_test() {
        // This test verifies that reshape with -1 (infer dimension) works with Shape inputs

        // Initialize the model
        let device = Default::default();
        let model: reshape_shape_with_neg::Model<TestBackend> =
            reshape_shape_with_neg::Model::new(&device);

        // Run the model with a tensor whose shape will be extracted and reshaped with -1
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 4], &device);
        let output = model.forward(input);

        // Output should be [2, 3, 4] reshaped to 1D with inferred size 3
        assert_eq!(output, [2i64, 3, 4]);
    }

    #[test]
    fn reshape_shape_partial_test() {
        // This test verifies partial reshaping of Shape arrays (Shape(2) -> Shape(2))

        // Initialize the model
        let device = Default::default();
        let model: reshape_shape_partial::Model<TestBackend> =
            reshape_shape_partial::Model::new(&device);

        // Run the model with a tensor whose shape will be sliced and reshaped
        let input = Tensor::<TestBackend, 4>::zeros([2, 3, 4, 5], &device);
        let output = model.forward(input);

        // Output should be [2, 3] - first two dimensions after slicing
        assert_eq!(output, [2i64, 3]);
    }

    #[test]
    fn reshape_scalar_to_scalar_test() {
        // This test verifies that Reshape(scalar, [-1]) keeps the output as scalar
        // instead of converting to a rank-1 tensor. This optimization avoids
        // wasteful tensor creation for single-element values.

        // Initialize the model
        let device = Default::default();
        let model: reshape_scalar_to_scalar::Model<TestBackend> =
            reshape_scalar_to_scalar::Model::new(&device);

        // Run the model with a 1x1 tensor input
        let input = Tensor::<TestBackend, 2>::from_floats([[42.5]], &device);
        let output = model.forward(input);

        // Output should be a scalar value (not a 1-element tensor)
        assert_eq!(output, 42.5f32);
    }
}

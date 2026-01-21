// Import the shared macro
use crate::include_models;
include_models!(
    expand,
    expand_scalar,
    expand_tensor,
    expand_shape,
    expand_with_where_shape,
    expand_max_semantics
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Shape, Tensor};

    use crate::backend::TestBackend;

    #[test]
    fn expand() {
        let device = Default::default();
        let model: expand::Model<TestBackend> = expand::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[-1.0], [1.0]], &device);

        let output = model.forward(input1);
        let expected_shape = Shape::from([2, 2]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_tensor() {
        let device = Default::default();
        let model: expand_tensor::Model<TestBackend> = expand_tensor::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[-1.0], [1.0]], &device);
        let input2 = Tensor::<TestBackend, 1, Int>::from_ints([2, 2], &device);

        let output = model.forward(input1, input2);
        let expected_shape = Shape::from([2, 2]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_scalar() {
        let device = Default::default();
        let model: expand_scalar::Model<TestBackend> = expand_scalar::Model::new(&device);

        let input = 5i64;
        let shape = Tensor::<TestBackend, 1, Int>::from_ints([2, 2], &device);

        let output = model.forward(input, shape);
        let expected_shape = Shape::from([2, 2]);

        assert_eq!(output.shape(), expected_shape);

        // Verify values: all elements should be 5
        let expected = Tensor::<TestBackend, 2, Int>::from_ints([[5, 5], [5, 5]], &device);
        output.into_data().assert_eq(&expected.into_data(), true);
    }

    #[test]
    fn expand_shape() {
        let device = Default::default();
        let model: expand_shape::Model<TestBackend> = expand_shape::Model::new(&device);

        let input1 = Tensor::<TestBackend, 2>::from_floats([[-1.0], [1.0], [1.0], [1.0]], &device);
        let input2 = Tensor::<TestBackend, 2>::zeros([4, 4], &device);

        let output = model.forward(input1, input2);
        let expected_shape = Shape::from([4, 4]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_with_where_shape() {
        let device = Default::default();
        // Use Model::default() to load constants from the record file
        let model: expand_with_where_shape::Model<TestBackend> =
            expand_with_where_shape::Model::default();

        // Input tensor to be expanded
        let input = Tensor::<TestBackend, 3>::ones([1, 1, 4], &device);

        // The model doesn't actually take condition as input - it's built into the model
        let output = model.forward(input);

        // The model has two constant shapes [2,3,4] selected by Where, then used in Expand
        // Result should be expanded to shape [2, 3, 4]
        let expected_shape = Shape::from([2, 3, 4]);

        assert_eq!(output.shape(), expected_shape);
    }

    #[test]
    fn expand_max_semantics() {
        // Tests ONNX Expand's max-semantics behavior:
        // When shape_dim=1 but input_dim>1, ONNX keeps the input_dim (not replaces with 1)
        // Input: [2, 3], Shape: [1, 1], Expected Output: [2, 3]
        let device = Default::default();
        let model: expand_max_semantics::Model<TestBackend> =
            expand_max_semantics::Model::new(&device);

        let input =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

        let output = model.forward(input.clone());

        // ONNX max-semantics: output_dim = max(input_dim, shape_dim)
        // max(2, 1) = 2, max(3, 1) = 3 => [2, 3]
        let expected_shape = Shape::from([2, 3]);
        assert_eq!(output.shape(), expected_shape);

        // Data should be preserved (no actual broadcasting occurred)
        output.into_data().assert_eq(&input.into_data(), true);
    }
}

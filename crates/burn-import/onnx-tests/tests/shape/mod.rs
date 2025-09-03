use crate::include_models;
include_models!(shape, shape_of_shape, shape_slice, shape_chain);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    use crate::backend::TestBackend;

    #[test]
    fn shape() {
        let device = Default::default();
        let model: shape::Model<TestBackend> = shape::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 2>::ones([4, 2], &device);
        let output = model.forward(input);
        let expected = [4i64, 2i64];
        assert_eq!(output, expected);
    }

    #[test]
    fn shape_of_shape() {
        let device = Default::default();
        let model: shape_of_shape::Model<TestBackend> = shape_of_shape::Model::new(&device);

        // Run the model - tests Shape operation on a Shape input
        let input = Tensor::<TestBackend, 4>::ones([2, 3, 4, 5], &device);
        let (shape1, shape2) = model.forward(input);

        // shape1 should be the shape of the input tensor
        let expected_shape1 = [2i64, 3i64, 4i64, 5i64];
        assert_eq!(shape1, expected_shape1);

        // shape2 should be the shape of shape1, which is just [4] (the rank)
        let expected_shape2 = [4i64];
        assert_eq!(shape2, expected_shape2);
    }

    #[test]
    fn shape_slice() {
        let device = Default::default();
        let model: shape_slice::Model<TestBackend> = shape_slice::Model::new(&device);

        // Run the model - tests Shape operation with start/end parameters
        let input = Tensor::<TestBackend, 5>::ones([2, 3, 4, 5, 6], &device);
        let output = model.forward(input);

        // With start=1, end=4, should extract dimensions [3, 4, 5]
        let expected = [3i64, 4i64, 5i64];
        assert_eq!(output, expected);
    }

    #[test]
    fn shape_chain() {
        let device = Default::default();
        let model: shape_chain::Model<TestBackend> = shape_chain::Model::new(&device);

        // Run the model - tests multiple chained Shape operations
        let input = Tensor::<TestBackend, 4>::ones([3, 4, 5, 6], &device);
        let (shape1, rank_shape, partial_shape, partial_rank_shape) = model.forward(input);

        // shape1 should be the full shape of the input
        assert_eq!(shape1, [3i64, 4i64, 5i64, 6i64]);

        // rank_shape should be [4] (the rank of the input)
        assert_eq!(rank_shape, [4i64]);

        // partial_shape should be [3, 4] (first 2 dims)
        assert_eq!(partial_shape, [3i64, 4i64]);

        // partial_rank_shape should be [2] (rank of partial shape)
        assert_eq!(partial_rank_shape, [2i64]);
    }
}

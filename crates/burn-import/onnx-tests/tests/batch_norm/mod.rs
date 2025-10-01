// Import the shared macro
use crate::include_models;
include_models!(batch_norm);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn batch_norm() {
        let model: batch_norm::Model<TestBackend> = batch_norm::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 3>::ones([1, 20, 1], &Default::default());
        let output = model.forward(input);

        let expected_shape = Shape::from([1, 5, 2, 2]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar();
        let expected_sum = 19.999_802; // from pytorch
        assert!(expected_sum.approx_eq(output_sum, (1.0e-8, 2)));
    }
}

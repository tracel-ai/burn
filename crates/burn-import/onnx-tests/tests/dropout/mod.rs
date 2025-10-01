// Import the shared macro
use crate::include_models;
include_models!(dropout);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn dropout() {
        let model: dropout::Model<TestBackend> = dropout::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 4, 10, 15]);
        assert_eq!(output.shape(), expected_shape);

        let output_sum = output.sum().into_scalar();

        let expected_sum = 1200.0; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }
}

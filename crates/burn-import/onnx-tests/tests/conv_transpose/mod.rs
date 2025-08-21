// Import the shared macro
use crate::include_models;
include_models!(conv_transpose1d, conv_transpose2d, conv_transpose3d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn conv_transpose1d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv_transpose1d::Model<TestBackend> = conv_transpose1d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 3>::ones([2, 4, 10], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 22]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv_transpose1d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = 33.810_33; // example result running the corresponding PyTorch model (conv_transpose1d.py)

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv_transpose2d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv_transpose2d::Model<TestBackend> = conv_transpose2d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 18, 15]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv_transpose2d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = -134.96603; // result running pytorch model (conv_transpose2d.py)

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv_transpose3d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv_transpose3d::Model<TestBackend> = conv_transpose3d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 5>::ones([2, 4, 4, 5, 7], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 6, 5, 9]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv_transpose3d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = -105.69771; // result running pytorch model (conv_transpose3d.py)

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }
}

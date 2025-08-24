// Import the shared macro
use crate::include_models;
include_models!(conv1d, conv2d, conv3d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use core::f64::consts;
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn conv1d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv1d::Model<TestBackend> = conv1d::Model::default();

        // Run the model with pi as input for easier testing
        let input = Tensor::<TestBackend, 3>::full([6, 4, 10], consts::PI, &Default::default());

        let output = model.forward(input);

        // test the output shape
        let expected_shape: Shape = Shape::from([6, 2, 7]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv1d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();
        let expected_sum = -54.549_243; // from pytorch
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv2d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv2d::Model<TestBackend> = conv2d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 6, 15]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv2d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = -113.869_99; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv3d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv3d::Model<TestBackend> = conv3d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<TestBackend, 5>::ones([2, 4, 4, 5, 7], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 3, 5, 5]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv3d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar();

        let expected_sum = 48.494_262; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }
}

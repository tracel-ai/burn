use crate::include_models;
include_models!(lstm);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn lstm_forward() {
        let device = Default::default();
        // Initialize the model with weights (loaded from the exported file)
        let model: lstm::Model<TestBackend> = lstm::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4]
        // Using random values matching the PyTorch test
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [1.9269, 1.4873, 0.9007, -2.1055],
                    [0.6784, -1.2345, -0.0431, -1.6047],
                ],
                [
                    [-0.7521, 1.6487, -0.3925, -1.4036],
                    [-0.7279, -0.5594, -0.7688, 0.7624],
                ],
                [
                    [1.6423, -0.1596, -0.4974, 0.4396],
                    [-0.7581, 1.0783, 0.8008, 1.6806],
                ],
                [
                    [1.2791, 1.2964, 0.6105, 1.3347],
                    [-0.2316, 0.0418, -0.2516, 0.8599],
                ],
                [
                    [-1.3847, -0.8712, -0.2234, 1.7174],
                    [0.3189, -0.4245, 0.3057, -0.7746],
                ],
            ],
            &device,
        );

        let (output, h_n, c_n) = model.forward(input);

        // Test output shapes
        // ONNX model has a Squeeze node that removes the num_directions dimension
        // Output: [seq_length, batch_size, hidden_size] = [5, 2, 8]
        let expected_output_shape = Shape::from([5, 2, 8]);
        // h_n: [num_directions, batch_size, hidden_size] = [1, 2, 8]
        let expected_h_shape = Shape::from([1, 2, 8]);
        // c_n: [num_directions, batch_size, hidden_size] = [1, 2, 8]
        let expected_c_shape = Shape::from([1, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);
        assert_eq!(c_n.shape(), expected_c_shape);

        // Verify approximate output values using sum (values from PyTorch)
        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();
        let c_n_sum = c_n.sum().into_scalar();

        // Expected values from ONNX runtime inference
        let expected_output_sum = -2.263_626_6;
        let expected_h_n_sum = -0.751_651_3;
        let expected_c_n_sum = -1.933_783_2;

        assert!(
            expected_output_sum.approx_eq(output_sum, (1.0e-4, 2)),
            "Output sum mismatch: expected {}, got {}",
            expected_output_sum,
            output_sum
        );
        assert!(
            expected_h_n_sum.approx_eq(h_n_sum, (1.0e-4, 2)),
            "h_n sum mismatch: expected {}, got {}",
            expected_h_n_sum,
            h_n_sum
        );
        assert!(
            expected_c_n_sum.approx_eq(c_n_sum, (1.0e-4, 2)),
            "c_n sum mismatch: expected {}, got {}",
            expected_c_n_sum,
            c_n_sum
        );
    }
}

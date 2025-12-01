use crate::include_models;
include_models!(
    lstm,
    lstm_bidirectional,
    lstm_reverse,
    lstm_with_initial_state
);

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

    #[test]
    fn lstm_bidirectional_forward() {
        let device = Default::default();
        let model: lstm_bidirectional::Model<TestBackend> = lstm_bidirectional::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4]
        // Using random values matching the PyTorch test (seed 42)
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-0.7111, -0.3867, 0.9578, -0.8225],
                    [-2.3908, 0.3222, 1.8754, 1.1043],
                ],
                [
                    [-0.5224, -0.7402, 0.1624, -0.2370],
                    [0.5099, 1.6706, 1.5921, -0.4162],
                ],
                [
                    [1.8619, -1.0779, 0.8849, -0.8342],
                    [1.0301, -0.8681, -0.5702, 0.3233],
                ],
                [
                    [0.7070, -1.2130, 0.8917, 0.7002],
                    [0.4082, -0.2351, 0.2602, -0.2455],
                ],
                [
                    [0.2245, -0.9745, 0.9817, 0.5837],
                    [0.2644, -0.8852, 0.7838, 0.8732],
                ],
            ],
            &device,
        );

        let (output, h_n, c_n) = model.forward(input);

        // Test output shapes for bidirectional LSTM
        // Output: [seq_length, batch_size, 2*hidden_size] = [5, 2, 16]
        let expected_output_shape = Shape::from([5, 2, 16]);
        // h_n: [num_directions, batch_size, hidden_size] = [2, 2, 8]
        let expected_h_shape = Shape::from([2, 2, 8]);
        // c_n: [num_directions, batch_size, hidden_size] = [2, 2, 8]
        let expected_c_shape = Shape::from([2, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);
        assert_eq!(c_n.shape(), expected_c_shape);

        // Verify approximate output values using sum (values from PyTorch)
        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();
        let c_n_sum = c_n.sum().into_scalar();

        // Expected values from PyTorch inference
        let expected_output_sum = 4.859_597_7;
        let expected_h_n_sum = 0.645_446_1;
        let expected_c_n_sum = 1.761_027_9;

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

    #[test]
    fn lstm_reverse_forward() {
        let device = Default::default();
        let model: lstm_reverse::Model<TestBackend> = lstm_reverse::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4]
        // Using random values matching the PyTorch test (seed 42)
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-1.3839, 0.4869, -1.0020, 0.0329],
                    [-0.4292, -0.9818, -0.6421, 0.8266],
                ],
                [
                    [1.5914, -0.1208, -0.4830, 0.1133],
                    [0.0772, -0.9228, -1.2620, 1.0861],
                ],
                [
                    [1.0966, -0.6837, 0.0660, -0.0008],
                    [0.1621, 1.1960, -1.3062, -1.4040],
                ],
                [
                    [-0.9711, 2.8605, -0.9941, -0.3590],
                    [-1.0009, -0.0631, -0.9607, 0.8314],
                ],
                [
                    [-2.3120, -0.6416, 1.0530, 1.4495],
                    [0.9528, 1.1891, -0.5683, -0.8502],
                ],
            ],
            &device,
        );

        let (output, h_n, c_n) = model.forward(input);

        // Test output shapes for reverse LSTM
        // Output: [seq_length, batch_size, hidden_size] = [5, 2, 8]
        // ONNX model has a Squeeze node that removes the num_directions dimension
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

        // Expected values from PyTorch inference with reverse direction simulation
        let expected_output_sum = -4.448_047_6;
        let expected_h_n_sum = -0.937_476_9;
        let expected_c_n_sum = -2.361_736_8;

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

    #[test]
    fn lstm_with_initial_state_forward() {
        let device = Default::default();
        let model: lstm_with_initial_state::Model<TestBackend> =
            lstm_with_initial_state::Model::default();

        // Input shape: [seq_length=5, batch_size=2, input_size=4]
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [-1.3839, 0.4869, -1.0020, 0.0329],
                    [-0.4292, -0.9818, -0.6421, 0.8266],
                ],
                [
                    [1.5914, -0.1208, -0.4830, 0.1133],
                    [0.0772, -0.9228, -1.2620, 1.0861],
                ],
                [
                    [1.0966, -0.6837, 0.0660, -0.0008],
                    [0.1621, 1.1960, -1.3062, -1.4040],
                ],
                [
                    [-0.9711, 2.8605, -0.9941, -0.3590],
                    [-1.0009, -0.0631, -0.9607, 0.8314],
                ],
                [
                    [-2.3120, -0.6416, 1.0530, 1.4495],
                    [0.9528, 1.1891, -0.5683, -0.8502],
                ],
            ],
            &device,
        );

        // Initial hidden state: [num_directions=1, batch_size=2, hidden_size=8]
        let h_0 = Tensor::<TestBackend, 3>::from_floats(
            [[
                [
                    -1.1766, 1.1889, 0.2156, 0.6265, 0.9144, 1.0095, 0.8946, -0.2878,
                ],
                [
                    -1.7459, 1.0964, 0.4866, 0.3668, -0.3912, 0.1699, -0.8704, 0.0305,
                ],
            ]],
            &device,
        );

        // Initial cell state: [num_directions=1, batch_size=2, hidden_size=8]
        let c_0 = Tensor::<TestBackend, 3>::from_floats(
            [[
                [
                    -0.8428, -1.6413, -0.7181, -0.1090, -1.8300, 0.3647, 0.7893, -1.3341,
                ],
                [
                    -0.6753, -0.8104, -0.8330, 1.8296, 0.7779, -0.0576, -1.3171, 0.1582,
                ],
            ]],
            &device,
        );

        let (output, h_n, c_n) = model.forward(input, h_0, c_0);

        // Test output shapes
        // ONNX model has a Squeeze node that removes the num_directions dimension
        let expected_output_shape = Shape::from([5, 2, 8]);
        let expected_h_shape = Shape::from([1, 2, 8]);
        let expected_c_shape = Shape::from([1, 2, 8]);

        assert_eq!(output.shape(), expected_output_shape);
        assert_eq!(h_n.shape(), expected_h_shape);
        assert_eq!(c_n.shape(), expected_c_shape);

        // Verify approximate output values using sum (values from PyTorch)
        let output_sum = output.sum().into_scalar();
        let h_n_sum = h_n.sum().into_scalar();
        let c_n_sum = c_n.sum().into_scalar();

        // Expected values from PyTorch inference with initial states
        let expected_output_sum = -7.564_189_4;
        let expected_h_n_sum = -1.002_073_3;
        let expected_c_n_sum = -2.692_845_8;

        // Use slightly looser tolerance for initial state test due to accumulated floating point differences
        assert!(
            expected_output_sum.approx_eq(output_sum, (1.0e-3, 2)),
            "Output sum mismatch: expected {}, got {}",
            expected_output_sum,
            output_sum
        );
        assert!(
            expected_h_n_sum.approx_eq(h_n_sum, (1.0e-3, 2)),
            "h_n sum mismatch: expected {}, got {}",
            expected_h_n_sum,
            h_n_sum
        );
        assert!(
            expected_c_n_sum.approx_eq(c_n_sum, (1.0e-3, 2)),
            "c_n sum mismatch: expected {}, got {}",
            expected_c_n_sum,
            c_n_sum
        );
    }
}

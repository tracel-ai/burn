// Tests for ONNX Scan operator

use crate::include_models;
include_models!(scan_cumsum, scan_reverse, scan_multi_state, scan_axis1);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn scan_cumsum_basic() {
        // Test cumulative sum with 1 state variable
        let device = Default::default();
        let model: scan_cumsum::Model<TestBackend> = Default::default();

        // Initial sum state [2, 3]
        let initial_sum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            &device,
        );

        // Input sequence [4, 2, 3] - 4 timesteps
        let input_sequence = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                [[2.0, 3.0, 1.0], [1.0, 2.0, 3.0]],
            ]),
            &device,
        );

        let (final_sum, cumsum_sequence) = model.forward(initial_sum, input_sequence);

        // Final sum should be sum of all inputs
        let expected_final = TensorData::from([[4.5, 6.5, 5.5], [6.5, 8.5, 10.5]]);
        final_sum
            .to_data()
            .assert_approx_eq::<f32>(&expected_final, burn::tensor::Tolerance::default());

        // Cumsum sequence shape should be [4, 2, 3]
        assert_eq!(cumsum_sequence.dims(), [4, 2, 3]);
    }

    #[test]
    fn scan_multi_state_lstm_like() {
        // Test with 2 state variables (hidden + cell)
        let device = Default::default();
        let model: scan_multi_state::Model<TestBackend> = Default::default();

        // Initial hidden state [2, 3]
        let initial_hidden = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            &device,
        );

        // Initial cell state [2, 3]
        let initial_cell = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            &device,
        );

        // Input sequence [4, 2, 3]
        let input_sequence = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]),
            &device,
        );

        let (final_hidden, final_cell, output_sequence) =
            model.forward(initial_hidden, initial_cell, input_sequence);

        // Both hidden and cell accumulate the inputs
        // After 4 steps: initial + sum of all inputs
        let expected_hidden = TensorData::from([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]);
        let expected_cell = TensorData::from([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);

        final_hidden
            .to_data()
            .assert_approx_eq::<f32>(&expected_hidden, burn::tensor::Tolerance::default());
        final_cell
            .to_data()
            .assert_approx_eq::<f32>(&expected_cell, burn::tensor::Tolerance::default());

        // Output sequence shape should be [4, 2, 3]
        assert_eq!(output_sequence.dims(), [4, 2, 3]);
    }

    #[test]
    fn scan_reverse_direction() {
        // Test reverse scanning with scan_input_directions=[1]
        let device = Default::default();
        let model: scan_reverse::Model<TestBackend> = Default::default();

        // Initial sum state [2, 3]
        let initial_sum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            &device,
        );

        // Input sequence [4, 2, 3] - will be processed in reverse (3→2→1→0)
        let input_sequence = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]),
            &device,
        );

        let (final_sum, cumsum_sequence) = model.forward(initial_sum, input_sequence);

        // Final sum should be sum of all inputs (order doesn't affect final sum)
        let expected_final = TensorData::from([[5.0, 2.0, 3.0], [0.0, 0.0, 0.0]]);
        final_sum
            .to_data()
            .assert_approx_eq::<f32>(&expected_final, burn::tensor::Tolerance::default());

        // Output sequence shape should be [4, 2, 3]
        assert_eq!(cumsum_sequence.dims(), [4, 2, 3]);
    }

    #[test]
    fn scan_axis1_cumsum() {
        // Test scanning along axis 1 (non-default scan axis)
        // NOTE: Expected values are manually computed because ONNX ReferenceEvaluator
        // doesn't support scan_input_axes != [0]
        let device = Default::default();
        let model: scan_axis1::Model<TestBackend> = Default::default();

        // Initial sum state [2, 2]
        let initial_sum = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 0.0], [0.0, 0.0]]),
            &device,
        );

        // Input sequence [2, 3, 2] (batch=2, seq=3, features=2)
        // Batch 0: [[1, 2], [3, 4], [5, 6]]
        // Batch 1: [[10, 20], [30, 40], [50, 60]]
        let input_sequence = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            ]),
            &device,
        );

        let (final_sum, cumsum_sequence) = model.forward(initial_sum, input_sequence);

        // Expected final sum:
        // Batch 0: [0,0] + [1,2] + [3,4] + [5,6] = [9, 12]
        // Batch 1: [0,0] + [10,20] + [30,40] + [50,60] = [90, 120]
        let expected_final = TensorData::from([[9.0, 12.0], [90.0, 120.0]]);
        final_sum
            .to_data()
            .assert_approx_eq::<f32>(&expected_final, burn::tensor::Tolerance::default());

        // Expected cumsum sequence:
        // Batch 0: [[1, 2], [4, 6], [9, 12]]
        // Batch 1: [[10, 20], [40, 60], [90, 120]]
        let expected_cumsum = TensorData::from([
            [[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]],
            [[10.0, 20.0], [40.0, 60.0], [90.0, 120.0]],
        ]);
        cumsum_sequence
            .to_data()
            .assert_approx_eq::<f32>(&expected_cumsum, burn::tensor::Tolerance::default());

        // Output sequence shape should be [2, 3, 2]
        assert_eq!(cumsum_sequence.dims(), [2, 3, 2]);
    }
}

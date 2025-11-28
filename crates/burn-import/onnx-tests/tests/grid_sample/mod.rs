// Import the shared macro
use crate::include_models;

// Include both grid_sample models
include_models!(grid_sample, grid_sample_nearest);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn grid_sample_bilinear() {
        // Test grid_sample with bilinear interpolation (default mode)
        let device = Default::default();
        let model = grid_sample::Model::<TestBackend>::new(&device);

        // Input: (1, 1, 4, 4)
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [0.3367, 0.1288, 0.2345, 0.2303],
                [-1.1229, -0.1863, 2.2082, -0.6380],
                [0.4617, 0.2674, 0.5349, 0.8094],
                [1.1103, -1.6898, -0.9890, 0.9580],
            ]]],
            &device,
        );

        // Grid: (1, 3, 3, 2) - normalized coordinates [-1, 1]
        let grid = Tensor::<TestBackend, 4>::from_floats(
            [[
                [[0.2961, -0.7596], [0.4927, 0.3031], [-0.0198, 0.9327]],
                [[-0.4663, -0.5019], [0.4681, 0.5351], [-0.4462, -0.5334]],
                [[0.3203, -0.7312], [-0.9372, -0.6755], [0.9645, 0.5409]],
            ]],
            &device,
        );

        let output = model.forward(input, grid);

        // Expected output shape: (1, 1, 3, 3)
        assert_eq!(output.dims(), [1, 1, 3, 3]);

        // Expected values from ONNX Runtime (verified with test_grid_sample.py)
        let expected = TensorData::from([[[
            [0.2296178f32, 0.5925206, -0.8675947],
            [-0.18328896, 0.20169558, -0.12067226],
            [0.29309627, 0.07458373, 0.51153356],
        ]]]);

        output.to_data().assert_approx_eq(
            &expected,
            burn::tensor::Tolerance::<f32>::rel_abs(1e-4, 1e-4),
        );
    }

    #[test]
    #[ignore = "nearest interpolation not yet implemented in burn-ndarray"]
    fn grid_sample_nearest() {
        // Test grid_sample with nearest neighbor interpolation
        let device = Default::default();
        let model = grid_sample_nearest::Model::<TestBackend>::new(&device);

        // Input: (1, 1, 4, 4)
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [0.3367, 0.1288, 0.2345, 0.2303],
                [-1.1229, -0.1863, 2.2082, -0.6380],
                [0.4617, 0.2674, 0.5349, 0.8094],
                [1.1103, -1.6898, -0.9890, 0.9580],
            ]]],
            &device,
        );

        // Grid: (1, 3, 3, 2) - normalized coordinates [-1, 1]
        let grid = Tensor::<TestBackend, 4>::from_floats(
            [[
                [[0.2961, -0.7596], [0.4927, 0.3031], [-0.0198, 0.9327]],
                [[-0.4663, -0.5019], [0.4681, 0.5351], [-0.4462, -0.5334]],
                [[0.3203, -0.7312], [-0.9372, -0.6755], [0.9645, 0.5409]],
            ]],
            &device,
        );

        let output = model.forward(input, grid);

        // Expected output shape: (1, 1, 3, 3)
        assert_eq!(output.dims(), [1, 1, 3, 3]);
    }
}

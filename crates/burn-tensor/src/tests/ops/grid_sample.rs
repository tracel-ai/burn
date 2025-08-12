#[burn_tensor_testgen::testgen(grid_sample)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    #[test]
    fn should_grid_sample_2d() {
        let device = Default::default();
        let tensor = TestTensor::<4>::from_floats(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
            &device,
        );
        let grid = TestTensor::<4>::from_floats(
            [[[[0.0, 0.0], [-1.0, 0.25]], [[1.0, 1.0], [0.2, -0.8]]]],
            &device,
        );

        let output = tensor.grid_sample_2d(grid);

        let expected = TensorData::from([[[[4.0, 3.75], [8.0, 1.8]]]]);
        output
            .to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    fn should_pad_border_grid_sample_2d() {
        let device = Default::default();
        let tensor = TestTensor::<4>::from_floats(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
            &device,
        );
        let grid = TestTensor::<4>::from_floats([[[[0.0, -2.0]]]], &device);

        let output = tensor.grid_sample_2d(grid);

        // Should clamp to nearest: 1.0
        let expected = TensorData::from([[[[1.0]]]]);
        output
            .to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }
}

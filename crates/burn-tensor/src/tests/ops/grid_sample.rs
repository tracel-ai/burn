#[burn_tensor_testgen::testgen(grid_sample)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_grid_sample_1d_dim0() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_floats([1.0, 0.0, 2.0], &device);
        let locations = TestTensor::from_floats(
            [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            &device,
        );

        let output = tensor.grid_sample_1d(0, locations);

        output.into_data().assert_eq(
            &TensorData::from([1.0, 0.75, 0.5, 0.25, 0.0, 0.5, 1.0, 1.5, 2.0]),
            false,
        );
    }

    // TODO: more tests
}

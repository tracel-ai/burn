// Import the shared macro
use crate::include_models;
include_models!(prelu, prelu_with_channel_slope);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn prelu() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: prelu::Model<TestBackend> = prelu::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.33669037, 0.0, 0.23446237],
                [0.23033303, -1.122_856, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [0.33669037f32, 0.0, 0.23446237],
            [0.23033303, -0.280714, -0.046582073],
        ]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn prelu_with_channel_slope() {
        let device = Default::default();
        let model: prelu_with_channel_slope::Model<TestBackend> =
            prelu_with_channel_slope::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats(
            [[
                [[0.5, -0.5], [1.0, -1.0]],   // ch0: mix of pos/neg
                [[0.1, 0.2], [0.3, 0.4]],     // ch1: all positive
                [[-0.1, -0.2], [-0.3, -0.4]], // ch2: all negative
            ]],
            &device,
        );

        let output = model.forward(input);

        let expected = TensorData::from([[
            [[0.5f32, -0.125], [1.0, -0.25]],  // ch0: neg scaled by 0.25
            [[0.1, 0.2], [0.3, 0.4]],          // ch1: unchanged
            [[-0.025, -0.05], [-0.075, -0.1]], // ch2: all scaled by 0.25
        ]]);
        output.to_data().assert_eq(&expected, true);
    }
}

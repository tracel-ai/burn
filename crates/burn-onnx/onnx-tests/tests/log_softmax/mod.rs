// Import the shared macro
use crate::include_models;
include_models!(log_softmax);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn log_softmax() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: log_softmax::Model<TestBackend> = log_softmax::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.33669037, 0.128_809_4, 0.23446237],
                [0.23033303, -1.122_856_4, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [-0.998_838_9f32, -1.206_719_9, -1.101_067],
            [-0.651_105_1, -2.004_294_6, -1.067_766_4],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}

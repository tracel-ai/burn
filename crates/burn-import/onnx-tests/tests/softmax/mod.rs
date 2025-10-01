// Import the shared macro
use crate::include_models;
include_models!(softmax);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn softmax() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: softmax::Model<TestBackend> = softmax::Model::new(&device);

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
            [0.36830685f32, 0.29917702, 0.33251613],
            [0.521_469_2, 0.13475533, 0.343_775_5],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}

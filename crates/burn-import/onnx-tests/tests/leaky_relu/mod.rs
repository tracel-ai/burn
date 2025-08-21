use crate::include_models;
include_models!(leaky_relu);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn leaky_relu() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: leaky_relu::Model<TestBackend> = leaky_relu::Model::new(&device);

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
            [0.23033303, -0.01122_856, -0.0018632829],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}

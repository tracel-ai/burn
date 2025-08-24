use crate::include_models;
include_models!(topk);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn topk() {
        // Initialize the model
        let device = Default::default();
        let model = topk::Model::<TestBackend>::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.33669037, 0.12880941, 0.23446237, 0.23033303, -1.12285638],
                [-0.18632829, 2.20820141, -0.63799703, 0.46165723, 0.26735088],
                [0.53490466, 0.80935723, 1.11029029, -1.68979895, -0.98895991],
            ],
            &device,
        );
        let (values_tensor, indices_tensor) = model.forward(input);

        // expected results
        let expected_values_tensor = TensorData::from([
            [0.33669037f32, 0.23446237],
            [2.208_201_4, 0.46165723],
            [1.110_290_3, 0.809_357_2],
        ]);
        let expected_indices_tensor = TensorData::from([[0i64, 2], [1, 3], [2, 1]]);

        values_tensor
            .to_data()
            .assert_eq(&expected_values_tensor, true);
        indices_tensor
            .to_data()
            .assert_eq(&expected_indices_tensor, true);
    }
}

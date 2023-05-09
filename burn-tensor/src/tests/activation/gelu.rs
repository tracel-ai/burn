#[burn_tensor_testgen::testgen(gelu)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data, Tensor};

    #[test]
    fn test_gelu() {
        let data = Data::from([[
            0.5447, 0.9809, 0.4114, 0.1398, 0.8045, 0.4103, 0.2388, 0.5262, 0.6677, 0.6737,
        ]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data).clone().clone();

        let data_actual = activation::gelu(tensor).to_data();

        let data_expected = Data::from([[
            0.3851, 0.8207, 0.2714, 0.0777, 0.6351, 0.2704, 0.1419, 0.3687, 0.4993, 0.5051,
        ]]);
        data_expected.assert_approx_eq(&data_actual, 2); // Low precision to allow approximation
                                                         // implementation using tanh
    }
}

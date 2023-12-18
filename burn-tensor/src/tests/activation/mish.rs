#[burn_tensor_testgen::testgen(mish)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data, Tensor};

    #[test]
    fn test_mish() {
        let data = Data::from([[-0.4240, -0.9574, -0.2215], [-0.5767, 0.7218, -0.1620]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = activation::mish(tensor).to_data();

        let data_expected = Data::from([[-0.1971, -0.3006, -0.1172], [-0.2413, 0.5823, -0.0888]]);
        data_actual.assert_approx_eq(&data_expected, 4);
    }
}

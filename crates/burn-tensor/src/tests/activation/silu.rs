#[burn_tensor_testgen::testgen(silu)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data, Tensor};

    #[test]
    fn test_silu() {
        let tensor = TestTensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let data_actual = activation::silu(tensor).into_data();

        let data_expected = Data::from([[0.7311, 1.7616], [2.8577, 3.9281]]);
        data_actual.assert_approx_eq(&data_expected, 4);
    }
}

#[burn_tensor_testgen::testgen(relu)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data, Tensor};

    #[test]
    fn test_relu_d2() {
        let data = Data::from([[0.0, -1.0, 2.0], [3.0, -4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = activation::relu(tensor).to_data();

        let data_expected = Data::from([[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]]);
        assert_eq!(data_expected, data_actual);
    }
}

#[burn_tensor_testgen::testgen(arg)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_argmax_2d() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.argmax(1);

        let data_expected = Data::from([[2], [2]]);
        assert_eq!(data_expected, data_actual.to_data());
    }
}

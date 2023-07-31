#[burn_tensor_testgen::testgen(clamp)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn clamp_min() {
        // test float tensor
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.clamp_min(2.0).into_data();

        let data_expected = Data::from([[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]]);
        assert_eq!(data_expected, data_actual);

        // test int tensor
        let data = Data::from([[0, 1, 2], [3, 4, 5]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data);
        let data_actual = tensor.clamp_min(2).into_data();
        let data_expected = Data::from([[2, 2, 2], [3, 4, 5]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn clamp_max() {
        // test float tensor
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.clamp_max(2.0).into_data();

        let data_expected = Data::from([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]]);
        assert_eq!(data_expected, data_actual);

        // test int tensor
        let data = Data::from([[0, 1, 2], [3, 4, 5]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data);
        let data_actual = tensor.clamp_max(4).into_data();
        let data_expected = Data::from([[0, 1, 2], [3, 4, 4]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn clamp_min_max() {
        // test float tensor
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);
        let data_actual = tensor.clamp(1.0, 4.0).into_data();
        let data_expected = Data::from([[1.0, 1.0, 2.0], [3.0, 4.0, 4.0]]);
        assert_eq!(data_expected, data_actual);

        // test int tensor
        let data = Data::from([[0, 1, 2], [3, 4, 5]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data);
        let data_actual = tensor.clamp(1, 4).into_data();
        let data_expected = Data::from([[1, 1, 2], [3, 4, 4]]);
        assert_eq!(data_expected, data_actual);
    }
}

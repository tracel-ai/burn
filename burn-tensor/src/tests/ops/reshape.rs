#[burn_tensor_testgen::testgen(reshape)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_reshape_1d() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data);

        let data_actual = tensor.clone().reshape([1, 3]).into_data();
        let data_expected = Data::from([[0.0, 1.0, 2.0]]);
        assert_eq!(data_expected, data_actual);

        // Infer the last dimension
        let data_actual = tensor.reshape([1, -1]).into_data();
        let data_expected = Data::from([[0.0, 1.0, 2.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_reshape_2d() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.clone().reshape([6]).into_data();
        let data_expected = Data::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(data_expected, data_actual);

        // Infer the dimension
        let data_actual = tensor.reshape([-1]).into_data();
        let data_expected = Data::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    #[should_panic]
    fn multiple_neg_ones() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data);
        let data_actual = tensor.reshape([-1, -1]).into_data();
    }

    #[test]
    #[should_panic]
    fn neg_value() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data);
        let data_actual = tensor.reshape([-2, -1]).into_data();
    }

    #[test]
    #[should_panic]
    fn zero_value() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data);
        let data_actual = tensor.reshape([0, -1]).into_data();
    }
}

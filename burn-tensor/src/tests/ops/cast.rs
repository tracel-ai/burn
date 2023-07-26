#[burn_tensor_testgen::testgen(cast)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn cast_float_tensor() {
        let float_data = Data::from([[1.0, 2.0, 3.0], [4.4, 5.5, 6.6]]);
        let float_tensor = Tensor::<TestBackend, 2>::from_data(float_data);

        let int_tensor = float_tensor.int();
        let data_expected = Data::from([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(data_expected, int_tensor.to_data());
    }

    #[test]
    fn cast_int_tensor() {
        let int_data = Data::from([[1, 2, 3], [4, 5, 6]]);
        let int_tensor = Tensor::<TestBackend, 2, Int>::from_data(int_data);

        let float_tensor = int_tensor.float();
        let data_expected = Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(data_expected, float_tensor.to_data());
    }
}

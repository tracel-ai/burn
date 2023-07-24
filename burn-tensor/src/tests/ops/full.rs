#[burn_tensor_testgen::testgen(full)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Shape, Tensor};

    #[test]
    fn test_data_full() {
        let data_actual = Data::full([2, 3].into(), 2.0);
        let data_expected = Data::from([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_tensor_full() {
        // Test full with f32
        let tensor = Tensor::<TestBackend, 2>::full([2, 3], 2.1);
        let data_expected = Data::from([[2.1, 2.1, 2.1], [2.1, 2.1, 2.1]]);
        assert_eq!(data_expected, tensor.into_data());

        // Test full with Int
        let int_tensor = Tensor::<TestBackend, 2, Int>::full([2, 2], 2);
        let data_expected = Data::from([[2, 2], [2, 2]]);
        assert_eq!(data_expected, int_tensor.into_data());
    }
}

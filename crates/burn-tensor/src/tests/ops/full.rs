#[burn_tensor_testgen::testgen(full)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Data, Int, Shape, Tensor};

    #[test]
    fn test_data_full() {
        let data_actual = Data::full([2, 3].into(), 2.0);
        let data_expected = Data::from([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_tensor_full() {
        let device = Default::default();
        // Test full with f32
        let tensor = Tensor::<TestBackend, 2>::full([2, 3], 2.1, &device);
        let data_expected = Data::from([[2.1, 2.1, 2.1], [2.1, 2.1, 2.1]]);
        assert_eq!(data_expected, tensor.into_data());

        // Test full with Int
        let int_tensor = Tensor::<TestBackend, 2, Int>::full([2, 2], 2, &device);
        let data_expected = Data::from([[2, 2], [2, 2]]);
        assert_eq!(data_expected, int_tensor.into_data());

        // TODO enable after adding support for bool
        // // Test full with bool
        // let bool_tensor = Tensor::<TestBackend, 2, Bool>::full([2, 2], true, &device);
        // let data_expected = Data::from([[true, true], [true, true]]);
        // assert_eq!(data_expected, bool_tensor.into_data());

        // let bool_tensor = Tensor::<TestBackend, 2, Bool>::full([2, 2], false, &device);
        // let data_expected = Data::from([[false, false], [false, false]]);
        // assert_eq!(data_expected, bool_tensor.into_data());
    }
}

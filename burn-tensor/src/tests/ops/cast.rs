#[burn_tensor_testgen::testgen(cast)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Data, Int, Tensor};

    #[test]
    fn cast_float_to_int() {
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.4, 5.5, 6.6]]);

        let actual = tensor.int().into_data();
        let expected = Data::from([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn cast_int_to_float_tensor() {
        let tensor = Tensor::<TestBackend, 2, Int>::from_data([[1, 2, 3], [4, 5, 6]]);

        let actual = tensor.float().into_data();
        let expected = Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn cast_bool_to_int_tensor() {
        let tensor =
            Tensor::<TestBackend, 2, Bool>::from_data([[true, false, true], [false, false, true]]);

        let actual = tensor.int().into_data();
        let expected = Data::from([[1, 0, 1], [0, 0, 1]]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn cast_bool_to_float_tensor() {
        let tensor =
            Tensor::<TestBackend, 2, Bool>::from_data([[true, false, true], [false, false, true]]);

        let actual = tensor.float().into_data();
        let expected = Data::from([[1., 0., 1.], [0., 0., 1.]]);
        assert_eq!(expected, actual);
    }
}

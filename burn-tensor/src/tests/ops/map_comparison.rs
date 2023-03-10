#[burn_tensor_testgen::testgen(map_comparison)]
mod tests {
    use super::*;
    use burn_tensor::{BasicOps, Data, Element, Float, Int, Numeric, Tensor, TensorKind};

    #[test]
    fn test_greater_elem() {
        greater_elem::<Float, f32>()
    }

    #[test]
    fn test_int_greater_elem() {
        greater_elem::<Int, i64>()
    }

    #[test]
    fn test_greater_equal_elem() {
        greater_equal_elem::<Float, f32>()
    }

    #[test]
    fn test_int_greater_equal_elem() {
        greater_equal_elem::<Float, f32>()
    }

    #[test]
    fn test_greater() {
        greater::<Float, f32>()
    }

    #[test]
    fn test_int_greater() {
        greater::<Int, i64>()
    }

    #[test]
    fn test_greater_equal() {
        greater_equal::<Float, f32>()
    }

    #[test]
    fn test_int_greater_equal() {
        greater_equal::<Int, i64>()
    }

    #[test]
    fn test_lower_elem() {
        lower_elem::<Float, f32>()
    }

    #[test]
    fn test_int_lower_elem() {
        lower_elem::<Int, i64>()
    }

    #[test]
    fn test_lower_equal_elem() {
        lower_equal_elem::<Float, f32>()
    }

    #[test]
    fn test_int_lower_equal_elem() {
        lower_equal_elem::<Int, i64>()
    }

    #[test]
    fn test_lower() {
        lower::<Float, f32>()
    }

    #[test]
    fn test_int_lower() {
        lower::<Int, i64>()
    }

    #[test]
    fn test_lower_equal() {
        lower_equal::<Float, f32>()
    }

    #[test]
    fn test_int_lower_equal() {
        lower_equal::<Int, i64>()
    }

    fn greater_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1: Data<<K as BasicOps<TestBackend>>::Elem, 2> =
            Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);

        let data_actual = tensor_1.greater_elem(4);

        let data_expected = Data::from([[false, false, false], [false, false, true]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    fn greater_equal_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);

        let data_actual = tensor_1.greater_equal_elem(4.0);

        let data_expected = Data::from([[false, false, false], [false, true, true]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    fn greater<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2>::from_data(data_2);

        let data_actual = tensor_1.greater(tensor_2);

        let data_expected = Data::from([[false, false, true], [false, true, false]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    fn greater_equal<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2>::from_data(data_2);

        let data_actual = tensor_1.greater_equal(tensor_2);

        let data_expected = Data::from([[false, true, true], [false, true, false]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    fn lower_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);

        let data_actual = tensor_1.lower_elem(4.0);

        let data_expected = Data::from([[true, true, true], [true, false, false]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    fn lower_equal_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);

        let data_actual = tensor_1.lower_equal_elem(4.0);

        let data_expected = Data::from([[true, true, true], [true, true, false]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    fn lower<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2>::from_data(data_2);

        let data_actual = tensor_1.lower(tensor_2);

        let data_expected = Data::from([[true, false, false], [true, false, true]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    fn lower_equal<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2>::from_data(data_2);

        let data_actual = tensor_1.lower_equal(tensor_2);

        let data_expected = Data::from([[true, true, false], [true, false, true]]);
        assert_eq!(data_expected, data_actual.to_data());
    }
}

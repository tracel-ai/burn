#[burn_tensor_testgen::testgen(map_comparison)]
mod tests {
    use super::*;
    use burn_tensor::{
        backend::Backend, BasicOps, Data, Element, Float, Int, Numeric, Tensor, TensorKind,
    };

    type IntElem = <TestBackend as Backend>::IntElem;
    type FloatElem = <TestBackend as Backend>::FloatElem;

    #[test]
    fn test_equal() {
        equal::<Float, FloatElem>()
    }

    #[test]
    fn test_int_equal() {
        equal::<Int, IntElem>()
    }

    #[test]
    fn test_equal_elem() {
        equal_elem::<Float, FloatElem>()
    }

    #[test]
    fn test_int_equal_elem() {
        equal_elem::<Int, IntElem>()
    }

    #[test]
    fn test_greater_elem() {
        greater_elem::<Float, FloatElem>()
    }

    #[test]
    fn test_int_greater_elem() {
        greater_elem::<Int, IntElem>()
    }

    #[test]
    fn test_greater_equal_elem() {
        greater_equal_elem::<Float, FloatElem>()
    }

    #[test]
    fn test_int_greater_equal_elem() {
        greater_equal_elem::<Float, FloatElem>()
    }

    #[test]
    fn test_greater() {
        greater::<Float, FloatElem>()
    }

    #[test]
    fn test_int_greater() {
        greater::<Int, IntElem>()
    }

    #[test]
    fn test_greater_equal() {
        greater_equal::<Float, FloatElem>()
    }

    #[test]
    fn test_int_greater_equal() {
        greater_equal::<Int, IntElem>()
    }

    #[test]
    fn test_lower_elem() {
        lower_elem::<Float, FloatElem>()
    }

    #[test]
    fn test_int_lower_elem() {
        lower_elem::<Int, IntElem>()
    }

    #[test]
    fn test_lower_equal_elem() {
        lower_equal_elem::<Float, FloatElem>()
    }

    #[test]
    fn test_int_lower_equal_elem() {
        lower_equal_elem::<Int, IntElem>()
    }

    #[test]
    fn test_lower() {
        lower::<Float, FloatElem>()
    }

    #[test]
    fn test_int_lower() {
        lower::<Int, IntElem>()
    }

    #[test]
    fn test_lower_equal() {
        lower_equal::<Float, FloatElem>()
    }

    #[test]
    fn test_int_lower_equal() {
        lower_equal::<Int, IntElem>()
    }

    fn equal<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert();
        let data_2 = Data::<f32, 2>::from([[1.0, 1.0, 1.0], [4.0, 3.0, 5.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2);

        let data_actual_cloned = tensor_1.clone().equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.equal(tensor_2);

        let data_expected = Data::from([[false, true, false], [false, false, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    fn equal_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 2.0, 5.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);

        let data_actual_cloned = tensor_1.clone().equal_elem(2);
        let data_actual_inplace = tensor_1.equal_elem(2);

        let data_expected = Data::from([[false, false, true], [false, true, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    fn greater_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);

        let data_actual_cloned = tensor_1.clone().greater_elem(4);
        let data_actual_inplace = tensor_1.greater_elem(4);

        let data_expected = Data::from([[false, false, false], [false, false, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    fn greater_equal_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);

        let data_actual_cloned = tensor_1.clone().greater_equal_elem(4.0);
        let data_actual_inplace = tensor_1.greater_equal_elem(4.0);

        let data_expected = Data::from([[false, false, false], [false, true, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    fn greater<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert();
        let data_2 = Data::<f32, 2>::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2);

        let data_actual_cloned = tensor_1.clone().greater(tensor_2.clone());
        let data_actual_inplace = tensor_1.greater(tensor_2);

        let data_expected = Data::from([[false, false, true], [false, true, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    fn greater_equal<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert();
        let data_2 = Data::<f32, 2>::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2);

        let data_actual_cloned = tensor_1.clone().greater_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.greater_equal(tensor_2);

        let data_expected = Data::from([[false, true, true], [false, true, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    fn lower_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);

        let data_actual_cloned = tensor_1.clone().lower_elem(4.0);
        let data_actual_inplace = tensor_1.lower_elem(4.0);

        let data_expected = Data::from([[true, true, true], [true, false, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    fn lower_equal_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);

        let data_actual_cloned = tensor_1.clone().lower_equal_elem(4.0);
        let data_actual_inplace = tensor_1.lower_equal_elem(4.0);

        let data_expected = Data::from([[true, true, true], [true, true, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    fn lower<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert();
        let data_2 = Data::<f32, 2>::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2);

        let data_actual_cloned = tensor_1.clone().lower(tensor_2.clone());
        let data_actual_inplace = tensor_1.lower(tensor_2);

        let data_expected = Data::from([[true, false, false], [true, false, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    fn lower_equal<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert();
        let data_2 = Data::<f32, 2>::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]).convert();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2);

        let data_actual_cloned = tensor_1.clone().lower_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.lower_equal(tensor_2);

        let data_expected = Data::from([[true, true, false], [true, false, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }
}

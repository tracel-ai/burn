#[burn_tensor_testgen::testgen(map_comparison)]
mod tests {
    use super::*;
    use burn_tensor::{
        backend::Backend, BasicOps, Bool, Element, Float, Int, Numeric, Tensor, TensorData,
        TensorKind,
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
    fn test_not_equal() {
        not_equal::<Float, FloatElem>()
    }

    #[test]
    fn test_int_not_equal() {
        not_equal::<Int, IntElem>()
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
    fn test_not_equal_elem() {
        not_equal_elem::<Float, FloatElem>()
    }

    #[test]
    fn test_int_not_equal_elem() {
        not_equal_elem::<Int, IntElem>()
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
        greater_equal_elem::<Int, IntElem>()
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

    #[test]
    fn test_equal_inf() {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [f32::INFINITY, 4.0, f32::NEG_INFINITY]]);
        let data_2 = TensorData::from([[1.0, 1.0, 1.0], [f32::INFINITY, 3.0, f32::NEG_INFINITY]]);
        let device = Default::default();
        let tensor_1 = TestTensor::<2>::from_data(data_1, &device);
        let tensor_2 = TestTensor::<2>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.equal(tensor_2);

        let data_expected = TensorData::from([[false, true, false], [true, false, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    #[test]
    fn test_not_equal_inf() {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, f32::INFINITY, 5.0]]);
        let data_2 = TensorData::from([[1.0, 1.0, 1.0], [f32::INFINITY, 3.0, f32::NEG_INFINITY]]);
        let device = Default::default();
        let tensor_1 = TestTensor::<2>::from_data(data_1, &device);
        let tensor_2 = TestTensor::<2>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().not_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.not_equal(tensor_2);

        let data_expected = TensorData::from([[true, false, true], [true, true, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn equal<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let data_2 = TensorData::from([[1.0, 1.0, 1.0], [4.0, 3.0, 5.0]]).convert::<E>();
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &device);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.equal(tensor_2);

        let data_expected = TensorData::from([[false, true, false], [false, false, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn not_equal<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let data_2 = TensorData::from([[1.0, 1.0, 1.0], [4.0, 3.0, 5.0]]).convert::<E>();
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &device);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().not_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.not_equal(tensor_2);

        let data_expected = TensorData::from([[true, false, true], [true, true, false]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn equal_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 2.0, 5.0]]).convert::<E>();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &Default::default());

        let data_actual_cloned = tensor_1.clone().equal_elem(2);
        let data_actual_inplace = tensor_1.equal_elem(2);

        let data_expected = TensorData::from([[false, false, true], [false, true, false]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn not_equal_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 2.0, 5.0]]).convert::<E>();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &Default::default());

        let data_actual_cloned = tensor_1.clone().not_equal_elem(2);
        let data_actual_inplace = tensor_1.not_equal_elem(2);

        let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn greater_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &Default::default());

        let data_actual_cloned = tensor_1.clone().greater_elem(4);
        let data_actual_inplace = tensor_1.greater_elem(4);

        let data_expected = TensorData::from([[false, false, false], [false, false, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn greater_equal_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &Default::default());

        let data_actual_cloned = tensor_1.clone().greater_equal_elem(4.0);
        let data_actual_inplace = tensor_1.greater_equal_elem(4.0);

        let data_expected = TensorData::from([[false, false, false], [false, true, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn greater<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let data_2 = TensorData::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]).convert::<E>();
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &device);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().greater(tensor_2.clone());
        let data_actual_inplace = tensor_1.greater(tensor_2);

        let data_expected = TensorData::from([[false, false, true], [false, true, false]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn greater_equal<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let data_2 = TensorData::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]).convert::<E>();
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &device);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().greater_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.greater_equal(tensor_2);

        let data_expected = TensorData::from([[false, true, true], [false, true, false]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn lower_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &Default::default());

        let data_actual_cloned = tensor_1.clone().lower_elem(4.0);
        let data_actual_inplace = tensor_1.lower_elem(4.0);

        let data_expected = TensorData::from([[true, true, true], [true, false, false]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn lower_equal_elem<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &Default::default());

        let data_actual_cloned = tensor_1.clone().lower_equal_elem(4.0);
        let data_actual_inplace = tensor_1.lower_equal_elem(4.0);

        let data_expected = TensorData::from([[true, true, true], [true, true, false]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn lower<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let data_2 = TensorData::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]).convert::<E>();
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &device);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().lower(tensor_2.clone());
        let data_actual_inplace = tensor_1.lower(tensor_2);

        let data_expected = TensorData::from([[true, false, false], [true, false, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    fn lower_equal<K, E>()
    where
        K: Numeric<TestBackend, Elem = E> + BasicOps<TestBackend, Elem = E>,
        E: Element,
    {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).convert::<E>();
        let data_2 = TensorData::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]).convert::<E>();
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2, K>::from_data(data_1, &device);
        let tensor_2 = Tensor::<TestBackend, 2, K>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().lower_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.lower_equal(tensor_2);

        let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    #[test]
    fn should_support_bool_equal() {
        let data_1 = TensorData::from([[false, true, true], [true, false, true]]);
        let data_2 = TensorData::from([[false, false, true], [false, true, true]]);
        let device = Default::default();
        let tensor_1 = TestTensorBool::<2>::from_data(data_1, &device);
        let tensor_2 = TestTensorBool::<2>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.equal(tensor_2);

        let data_expected = TensorData::from([[true, false, true], [false, false, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    #[test]
    fn should_support_bool_not_equal() {
        let data_1 = TensorData::from([[false, true, true], [true, false, true]]);
        let data_2 = TensorData::from([[false, false, true], [false, true, true]]);
        let device = Default::default();
        let tensor_1 = TestTensorBool::<2>::from_data(data_1, &device);
        let tensor_2 = TestTensorBool::<2>::from_data(data_2, &device);

        let data_actual_cloned = tensor_1.clone().not_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.not_equal(tensor_2);

        let data_expected = TensorData::from([[false, true, false], [true, true, false]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }

    #[test]
    fn should_support_bool_not() {
        let data_1 = TensorData::from([[false, true, true], [true, true, false]]);
        let tensor_1 = TestTensorBool::<2>::from_data(data_1, &Default::default());

        let data_actual_cloned = tensor_1.clone().bool_not();
        let data_actual_inplace = tensor_1.bool_not();

        let data_expected = TensorData::from([[true, false, false], [false, false, true]]);
        data_expected.assert_eq(&data_actual_cloned.into_data(), false);
        data_expected.assert_eq(&data_actual_inplace.into_data(), false);
    }
}

#[burn_tensor_testgen::testgen(one_hot)]
mod tests {
    use super::*;
    use burn_tensor::{Int, TensorData};

    #[test]
    fn float_should_support_one_hot() {
        let device = Default::default();

        let tensor = TestTensor::<1>::one_hot(0, 5, &device);
        let expected = TensorData::from([1., 0., 0., 0., 0.]);
        tensor.into_data().assert_eq(&expected, false);

        let tensor = TestTensor::<1>::one_hot(1, 5, &device);
        let expected = TensorData::from([0., 1., 0., 0., 0.]);
        tensor.into_data().assert_eq(&expected, false);

        let tensor = TestTensor::<1>::one_hot(4, 5, &device);
        let expected = TensorData::from([0., 0., 0., 0., 1.]);
        tensor.into_data().assert_eq(&expected, false);

        let tensor = TestTensor::<1>::one_hot(1, 2, &device);
        let expected = TensorData::from([0., 1.]);
        tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic]
    fn float_one_hot_should_panic_when_index_exceeds_number_of_classes() {
        let device = Default::default();
        let tensor = TestTensor::<1>::one_hot(1, 1, &device);
    }

    #[test]
    #[should_panic]
    fn float_one_hot_should_panic_when_number_of_classes_is_zero() {
        let device = Default::default();
        let tensor = TestTensor::<1>::one_hot(0, 0, &device);
    }

    #[test]
    fn int_should_support_one_hot() {
        let device = Default::default();

        let index_tensor = TestTensorInt::<1>::arange(0..5, &device);
        let one_hot_tensor = index_tensor.one_hot(5);
        let expected = TestTensorInt::eye(5, &device).into_data();
        one_hot_tensor.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic]
    fn int_one_hot_should_panic_when_index_exceeds_number_of_classes() {
        let device = Default::default();
        let index_tensor = TestTensorInt::<1>::arange(0..6, &device);
        let one_hot_tensor = index_tensor.one_hot(5);
    }

    #[test]
    #[should_panic]
    fn int_one_hot_should_panic_when_number_of_classes_is_zero() {
        let device = Default::default();
        let index_tensor = TestTensorInt::<1>::arange(0..3, &device);
        let one_hot_tensor = index_tensor.one_hot(0);
    }

    #[test]
    #[should_panic]
    fn int_one_hot_should_panic_when_number_of_classes_is_1() {
        let device = Default::default();
        let index_tensor = TestTensorInt::<1>::arange(0..3, &device);
        let one_hot_tensor = index_tensor.one_hot(1);
    }

    #[test]
    #[should_panic]
    fn int_one_hot_with() {
        let device = Default::default();
        let index_tensor = TestTensorInt::<1>::arange(0..3, &device);
        let expected = TestTensorInt::eye(5, &device).into_data();

        let one_hot_tensor = index_tensor.one_hot_with_axis_and_values(3, 5, 0, -1);
        
        one_hot_tensor.into_data().assert_eq(&expected, false);
    }
}

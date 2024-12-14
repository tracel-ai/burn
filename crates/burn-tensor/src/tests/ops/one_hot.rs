#[burn_tensor_testgen::testgen(one_hot)]
mod tests {
    use super::*;
    use burn_tensor::{
        as_type,
        backend::Backend,
        tests::{Float as _, Int as _},
        Float, Int, Numeric, Shape, Tensor, TensorData,
    };
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
    fn one_hot_plus_with_positive_axis_and_indices() {
        let tensor = TestTensorInt::<2>::from([[1, 9], [2, 4]]);
        let expected = TensorData::from(as_type!(IntType:[
           [[1, 1],
            [3, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 3]],
           [[1, 1],
            [1, 1],
            [3, 1],
            [1, 1],
            [1, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1]]]));

        let one_hot_tensor: Tensor<TestBackend, 3, Int> = tensor.one_hot_plus(10, 3, 1, 1);

        one_hot_tensor.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_plus_with_negative_axis_and_indices() {
        let tensor = TestTensorInt::<2>::from([[0, 2], [1, -1]]);
        let expected = TensorData::from(
            as_type!(FloatType: [[[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]], [[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]]),
        );

        let one_hot_tensor: Tensor<TestBackend, 3> =
            tensor.one_hot_plus(3, FloatType::new(5.0), FloatType::new(0.0), -1);

        one_hot_tensor.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_plus_with_negative_indices() {
        let tensor = TestTensor::<1>::from([0.0, -7.0, -8.0]);
        let expected = TensorData::from(as_type!(FloatType:[
            [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]));

        let one_hot_tensor: Tensor<TestBackend, 2> =
            tensor.one_hot_plus(10, 3.0, 1.0, 1);

        one_hot_tensor.into_data().assert_eq(&expected, true);
    }

    #[should_panic]
    #[test]
    fn one_hot_plus_should_panic_when_axis_out_range_of_rank() {
        let tensor = TestTensor::<2>::from([[0.0, 2.0], [1.0, -1.0]]);

        let one_hot_tensor: Tensor<TestBackend, 3, Float> =
            tensor.one_hot_plus(3, FloatType::new(5.0), FloatType::new(0.0), 3);
    }
}

#[burn_tensor_testgen::testgen(one_hot)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int};

    #[test]
    fn should_support_one_hot() {
        let device = Default::default();

        let tensor = TestTensor::<1>::one_hot(0, 5, &device);
        assert_eq!(tensor.to_data(), Data::from([1., 0., 0., 0., 0.]));

        let tensor = TestTensor::<1>::one_hot(1, 5, &device);
        assert_eq!(tensor.to_data(), Data::from([0., 1., 0., 0., 0.]));

        let tensor = TestTensor::<1>::one_hot(4, 5, &device);
        assert_eq!(tensor.to_data(), Data::from([0., 0., 0., 0., 1.]));

        let tensor = TestTensor::<1>::one_hot(1, 2, &device);
        assert_eq!(tensor.to_data(), Data::from([0., 1.]));
    }

    #[test]
    #[should_panic]
    fn should_panic_when_index_exceeds_number_of_classes() {
        let device = Default::default();
        let tensor = TestTensor::<1>::one_hot(1, 1, &device);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_number_of_classes_is_zero() {
        let device = Default::default();
        let tensor = TestTensor::<1>::one_hot(0, 0, &device);
    }
}

#[burn_tensor_testgen::testgen(expand)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Shape, Tensor};

    #[test]
    fn expand_2d() {
        let tensor = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &Default::default());
        let broadcasted_tensor = tensor.expand([3, 3]);

        let expected_data = Data::from([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
        assert_eq!(broadcasted_tensor.into_data(), expected_data);

        let tensor =
            Tensor::<TestBackend, 1>::from_floats([4.0, 7.0, 2.0, 3.0], &Default::default());
        let broadcasted_tensor = tensor.expand([2, 4]);

        let expected_data = Data::from([[4.0, 7.0, 2.0, 3.0], [4.0, 7.0, 2.0, 3.0]]);
        assert_eq!(broadcasted_tensor.into_data(), expected_data);
    }

    #[test]
    fn expand_3d() {
        let tensor =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &Default::default());
        let broadcasted_tensor = tensor.expand([3, 2, 2]);

        let expected_data = Data::from([
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ]);
        assert_eq!(broadcasted_tensor.into_data(), expected_data);
    }

    #[test]
    fn expand_higher_dimensions() {
        let tensor =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &Default::default());
        let broadcasted_tensor = tensor.expand([2, 3, 4]);

        let expected_data = Data::from([
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
        ]);

        assert_eq!(broadcasted_tensor.into_data(), expected_data);
    }

    #[test]
    fn broadcast_single() {
        let tensor = Tensor::<TestBackend, 1>::from_floats([1.0], &Default::default());
        let broadcasted_tensor = tensor.expand([2, 3]);

        let expected_data = Data::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        assert_eq!(broadcasted_tensor.into_data(), expected_data);
    }

    #[test]
    #[should_panic]
    fn should_fail_expand_incompatible_shapes() {
        let tensor = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &Default::default());
        let _broadcasted_tensor = tensor.expand([2, 2]);
    }

    #[test]
    fn expand_2d_bool() {
        let tensor = TestTensorBool::from([false, true, false]);
        let broadcasted_tensor = tensor.expand([3, 3]);

        let expected_data = Data::from([
            [false, true, false],
            [false, true, false],
            [false, true, false],
        ]);
        assert_eq!(broadcasted_tensor.into_data(), expected_data);
    }

    #[test]
    fn expand_2d_int() {
        let tensor = TestTensorInt::from([1, 2, 3]);
        let broadcasted_tensor = tensor.expand([3, 3]);

        let expected_data = Data::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]]);
        assert_eq!(broadcasted_tensor.into_data(), expected_data);
    }

    #[test]
    fn should_all_negative_one() {
        let tensor = TestTensorInt::from([1, 2, 3]);
        let broadcasted_tensor = tensor.expand([2, -1]);

        let expected_data = Data::from([[1, 2, 3], [1, 2, 3]]);
        assert_eq!(broadcasted_tensor.into_data(), expected_data);
    }

    #[test]
    #[should_panic]
    fn should_panic_negative_one_on_non_existing_dim() {
        let tensor = TestTensorInt::from([1, 2, 3]);

        println!("{:?}", tensor.shape());
        let _broadcasted_tensor = tensor.expand([-1, 3]);
        println!("{:?}", _broadcasted_tensor.shape());
    }
}

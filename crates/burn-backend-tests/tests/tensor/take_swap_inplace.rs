mod float {
    use crate::TestTensor;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_release_swap() {
        let mut tensor: Tensor<1> =
            TestTensor::<1>::from_data(TensorData::from([0.0, 1.0, 2.0, 3.0]), &Default::default());
        assert_eq!(tensor.dims(), [4]);

        let mut old: Tensor<1> = tensor.take();
        assert_eq!(tensor.dims(), [0]);
        assert_eq!(old.dims(), [4]);

        tensor.swap(&mut old);
        assert_eq!(tensor.dims(), [4]);
        assert_eq!(old.dims(), [0]);
    }

    #[test]
    fn test_inplace() {
        let mut tensor: Tensor<1> =
            TestTensor::<1>::from_data(TensorData::from([0.0, 1.0, 2.0, 3.0]), &Default::default());
        tensor.inplace(|t| t.square());

        tensor
            .to_data()
            .assert_eq(&TensorData::from([0.0, 1.0, 4.0, 9.0]), false);
    }
}

mod int {
    use crate::TestTensorInt;
    use burn_tensor::{Int, Tensor, TensorData};

    #[test]
    fn test_release_swap() {
        let mut tensor: Tensor<1, Int> =
            TestTensorInt::<1>::from_data(TensorData::from([0, 1, 2, 3]), &Default::default());
        assert_eq!(tensor.dims(), [4]);

        let mut old: Tensor<1, Int> = tensor.take();
        assert_eq!(tensor.dims(), [0]);
        assert_eq!(old.dims(), [4]);

        tensor.swap(&mut old);
        assert_eq!(tensor.dims(), [4]);
        assert_eq!(old.dims(), [0]);
    }

    #[test]
    fn test_inplace() {
        let mut tensor: Tensor<1, Int> =
            TestTensorInt::<1>::from_data(TensorData::from([0, 1, 2, 3]), &Default::default());
        tensor.inplace(|t| t.clone() * t);

        tensor
            .to_data()
            .assert_eq(&TensorData::from([0, 1, 4, 9]), false);
    }
}

mod bool {
    use crate::TestTensorBool;
    use burn_tensor::{Bool, Tensor, TensorData};

    #[test]
    fn test_release_swap() {
        let mut tensor: Tensor<1, Bool> = TestTensorBool::<1>::from_data(
            TensorData::from([true, true, false, false]),
            &Default::default(),
        );
        assert_eq!(tensor.dims(), [4]);

        let mut old: Tensor<1, Bool> = tensor.take();
        assert_eq!(tensor.dims(), [0]);
        assert_eq!(old.dims(), [4]);

        tensor.swap(&mut old);
        assert_eq!(tensor.dims(), [4]);
        assert_eq!(old.dims(), [0]);
    }

    #[test]
    fn test_inplace() {
        let mut tensor: Tensor<1, Bool> = TestTensorBool::<1>::from_data(
            TensorData::from([true, true, false, false]),
            &Default::default(),
        );
        tensor.inplace(|t| t.bool_not());

        tensor
            .to_data()
            .assert_eq(&TensorData::from([false, false, true, true]), false);
    }
}

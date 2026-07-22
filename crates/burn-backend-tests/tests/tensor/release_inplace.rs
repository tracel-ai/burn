mod float {
    use crate::TestTensor;
    use burn_tensor::TensorData;

    #[test]
    fn test_release_swap() {
        let mut tensor =
            TestTensor::<1>::from_data(TensorData::from([0, 1, 2, 3]), &Default::default());
        assert_eq!(tensor.dims(), [4]);

        let mut old = tensor.release();
        assert_eq!(tensor.dims(), [0]);
        assert_eq!(old.dims(), [4]);

        tensor.swap(&mut old);
        assert_eq!(tensor.dims(), [4]);
        assert_eq!(old.dims(), [0]);
    }

    #[test]
    fn test_inplace() {
        let mut tensor =
            TestTensor::<1>::from_data(TensorData::from([0, 1, 2, 3]), &Default::default());
        tensor.inplace(|t| t.square());

        tensor
            .to_data()
            .assert_eq(&TensorData::from([0, 1, 4, 9]), false);
    }
}

#[burn_tensor_testgen::testgen(argwhere_nonzero)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{backend::Backend, Tensor, TensorData};

    #[test]
    fn test_argwhere_1d() {
        let tensor = TestTensorBool::<1>::from([false, true, false, true, true]);
        let output = tensor.argwhere();
        let expected =
            TensorData::from([[1], [3], [4]]).convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_argwhere_2d() {
        let tensor = TestTensorBool::<2>::from([[false, false], [false, true], [true, true]]);
        let output = tensor.argwhere();
        let expected = TensorData::from([[1, 1], [2, 0], [2, 1]])
            .convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_argwhere_3d() {
        let tensor = TestTensorBool::<3>::from([
            [[false, false, false], [false, true, false]],
            [[true, false, true], [true, true, false]],
        ]);
        let output = tensor.argwhere();
        let expected = TensorData::from([[0, 1, 1], [1, 0, 0], [1, 0, 2], [1, 1, 0], [1, 1, 1]])
            .convert::<<TestBackend as Backend>::IntElem>();

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn test_nonzero_1d() {
        let tensor = TestTensorBool::<1>::from([false, true, false, true, true]);
        let data_actual = tensor
            .nonzero()
            .into_iter()
            .map(|t| t.into_data())
            .collect::<Vec<_>>();
        let data_expected =
            vec![TensorData::from([1, 3, 4]).convert::<<TestBackend as Backend>::IntElem>()];
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_nonzero_2d() {
        // 2-D tensor
        let tensor = TestTensorBool::<2>::from([[false, false], [false, true], [true, true]]);
        let data_actual = tensor
            .nonzero()
            .into_iter()
            .map(|t| t.into_data())
            .collect::<Vec<_>>();
        let data_expected = vec![
            TensorData::from([1, 2, 2]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([1, 0, 1]).convert::<<TestBackend as Backend>::IntElem>(),
        ];
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_nonzero_3d() {
        // 3-D tensor
        let tensor = TestTensorBool::<3>::from([
            [[false, false, false], [false, true, false]],
            [[true, false, true], [true, true, false]],
        ]);
        let data_actual = tensor
            .nonzero()
            .into_iter()
            .map(|t| t.into_data())
            .collect::<Vec<_>>();
        let data_expected = vec![
            TensorData::from([0, 1, 1, 1, 1]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([1, 0, 0, 1, 1]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([1, 0, 2, 0, 1]).convert::<<TestBackend as Backend>::IntElem>(),
        ];
        assert_eq!(data_expected, data_actual);
    }
}

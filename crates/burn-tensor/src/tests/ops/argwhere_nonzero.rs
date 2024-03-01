#[burn_tensor_testgen::testgen(argwhere_nonzero)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_argwhere_1d() {
        // 1-D tensor
        let tensor = TestTensorBool::from([false, true, false, true, true]);
        let data_actual = tensor.argwhere().into_data();
        let data_expected = Data::from([[1], [3], [4]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_argwhere_2d() {
        // 2-D tensor
        let tensor = TestTensorBool::from([[false, false], [false, true], [true, true]]);
        let data_actual = tensor.argwhere().into_data();
        // let data_expected = vec![Data::from([1, 2, 2]), Data::from([1, 0, 1])];
        let data_expected = Data::from([[1, 1], [2, 0], [2, 1]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_argwhere_3d() {
        // 3-D tensor
        let tensor = TestTensorBool::from([
            [[false, false, false], [false, true, false]],
            [[true, false, true], [true, true, false]],
        ]);
        let data_actual = tensor.argwhere().into_data();
        let data_expected = Data::from([[0, 1, 1], [1, 0, 0], [1, 0, 2], [1, 1, 0], [1, 1, 1]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_nonzero_1d() {
        // 1-D tensor
        let tensor = TestTensorBool::from([false, true, false, true, true]);
        let data_actual = tensor
            .nonzero()
            .into_iter()
            .map(|t| t.into_data())
            .collect::<Vec<_>>();
        let data_expected = vec![Data::from([1, 3, 4])];
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_nonzero_2d() {
        // 2-D tensor
        let tensor = TestTensorBool::from([[false, false], [false, true], [true, true]]);
        let data_actual = tensor
            .nonzero()
            .into_iter()
            .map(|t| t.into_data())
            .collect::<Vec<_>>();
        let data_expected = vec![Data::from([1, 2, 2]), Data::from([1, 0, 1])];
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_nonzero_3d() {
        // 3-D tensor
        let tensor = TestTensorBool::from([
            [[false, false, false], [false, true, false]],
            [[true, false, true], [true, true, false]],
        ]);
        let data_actual = tensor
            .nonzero()
            .into_iter()
            .map(|t| t.into_data())
            .collect::<Vec<_>>();
        let data_expected = vec![
            Data::from([0, 1, 1, 1, 1]),
            Data::from([1, 0, 0, 1, 1]),
            Data::from([1, 0, 2, 0, 1]),
        ];
        assert_eq!(data_expected, data_actual);
    }
}

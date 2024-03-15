#[burn_tensor_testgen::testgen(tri_mask)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Data, Tensor};

    #[test]
    fn square_diag() {
        let device = Default::default();
        let data_expected = Data::from([
            [false, true, true],
            [true, false, true],
            [true, true, false],
        ]);
        let tensor = Tensor::<TestBackend, 2, Bool>::diag_mask([3, 3], 0, &device);
        assert_eq!(data_expected, tensor.into_data());
    }

    #[test]
    fn square_diag_offset() {
        let device = Default::default();
        let data_expected =
            Data::from([[true, false, true], [true, true, false], [true, true, true]]);
        let tensor = Tensor::<TestBackend, 2, Bool>::diag_mask([3, 3], 1, &device);
        assert_eq!(data_expected, tensor.into_data());
    }

    #[test]
    fn square_tri_upper() {
        let device = Default::default();
        let data_expected = Data::from([
            [false, false, false],
            [true, false, false],
            [true, true, false],
        ]);
        let tensor = Tensor::<TestBackend, 2, Bool>::triu_mask([3, 3], 0, &device);
        assert_eq!(data_expected, tensor.into_data());
    }

    #[test]
    fn square_tri_upper_offset() {
        let device = Default::default();
        let data_expected = Data::from([
            [true, false, false],
            [true, true, false],
            [true, true, true],
        ]);
        let tensor = Tensor::<TestBackend, 2, Bool>::triu_mask([3, 3], 1, &device);
        assert_eq!(data_expected, tensor.into_data());
    }

    #[test]
    fn square_tri_lower() {
        let device = Default::default();

        let data_expected = Data::from([
            [false, true, true],
            [false, false, true],
            [false, false, false],
        ]);
        let tensor = Tensor::<TestBackend, 2, Bool>::tril_mask([3, 3], 0, &device);
        assert_eq!(data_expected, tensor.into_data());
    }

    #[test]
    fn square_tri_lower_offset() {
        let device = Default::default();

        let data_expected = Data::from([
            [true, true, true],
            [false, true, true],
            [false, false, true],
        ]);
        let tensor = Tensor::<TestBackend, 2, Bool>::tril_mask([3, 3], -1, &device);
        assert_eq!(data_expected, tensor.into_data());
    }

    #[test]
    fn rect_diag() {
        let device = Default::default();
        let data_expected = Data::from([
            [false, true, true, true],
            [true, false, true, true],
            [true, true, false, true],
        ]);
        let tensor = Tensor::<TestBackend, 2, Bool>::diag_mask([3, 4], 0, &device);
        assert_eq!(data_expected, tensor.into_data());

        let data_expected = Data::from([
            [false, true, true],
            [true, false, true],
            [true, true, false],
            [true, true, true],
        ]);
        let tensor = Tensor::<TestBackend, 2, Bool>::diag_mask([4, 3], 0, &device);
        assert_eq!(data_expected, tensor.into_data());
    }
}

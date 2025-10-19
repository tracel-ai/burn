#[burn_tensor_testgen::testgen(tri_mask)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Shape, Tensor, TensorData};

    #[test]
    fn square_diag() {
        let device = Default::default();
        let data_expected = TensorData::from([
            [false, true, true],
            [true, false, true],
            [true, true, false],
        ]);
        let tensor = TestTensorBool::<2>::diag_mask([3, 3], 0, &device);
        tensor.into_data().assert_eq(&data_expected, false);
    }

    #[test]
    fn square_diag_offset() {
        let device = Default::default();
        let data_expected =
            TensorData::from([[true, false, true], [true, true, false], [true, true, true]]);
        let tensor = TestTensorBool::<2>::diag_mask([3, 3], 1, &device);
        tensor.into_data().assert_eq(&data_expected, false);
    }

    #[test]
    fn square_tri_upper() {
        let device = Default::default();
        let data_expected = TensorData::from([
            [false, false, false],
            [true, false, false],
            [true, true, false],
        ]);
        let tensor = TestTensorBool::<2>::triu_mask([3, 3], 0, &device);
        tensor.into_data().assert_eq(&data_expected, false);
    }

    #[test]
    fn square_tri_upper_offset() {
        let device = Default::default();
        let data_expected = TensorData::from([
            [true, false, false],
            [true, true, false],
            [true, true, true],
        ]);
        let tensor = TestTensorBool::<2>::triu_mask([3, 3], 1, &device);
        tensor.into_data().assert_eq(&data_expected, false);
    }

    #[test]
    fn square_tri_lower() {
        let device = Default::default();

        let data_expected = TensorData::from([
            [false, true, true],
            [false, false, true],
            [false, false, false],
        ]);
        let tensor = TestTensorBool::<2>::tril_mask([3, 3], 0, &device);
        tensor.into_data().assert_eq(&data_expected, false);
    }

    #[test]
    fn square_tri_lower_offset() {
        let device = Default::default();

        let data_expected = TensorData::from([
            [true, true, true],
            [false, true, true],
            [false, false, true],
        ]);
        let tensor = TestTensorBool::<2>::tril_mask([3, 3], -1, &device);
        tensor.into_data().assert_eq(&data_expected, false);
    }

    #[test]
    fn rect_diag() {
        let device = Default::default();
        let data_expected = TensorData::from([
            [false, true, true, true],
            [true, false, true, true],
            [true, true, false, true],
        ]);
        let tensor = TestTensorBool::<2>::diag_mask([3, 4], 0, &device);
        tensor.into_data().assert_eq(&data_expected, false);

        let data_expected = TensorData::from([
            [false, true, true],
            [true, false, true],
            [true, true, false],
            [true, true, true],
        ]);
        let tensor = TestTensorBool::<2>::diag_mask([4, 3], 0, &device);
        tensor.into_data().assert_eq(&data_expected, false);
    }

    #[test]
    fn triu_mask_matches_batched_shape() {
        let device = Default::default();
        let shape = [2, 3, 4];

        let mask = TestTensorBool::<3>::triu_mask(shape, 0, &device);

        assert_eq!(mask.shape(), Shape::new(shape));
    }

    #[test]
    fn tril_mask_matches_batched_shape() {
        let device = Default::default();
        let shape = [3, 2, 5];

        let mask = TestTensorBool::<3>::tril_mask(shape, 0, &device);

        assert_eq!(mask.shape(), Shape::new(shape));
    }

    #[test]
    fn diag_mask_matches_batched_shape() {
        let device = Default::default();
        let shape = [4, 2, 2, 3];

        let mask = TestTensorBool::<4>::diag_mask(shape, 0, &device);

        assert_eq!(mask.shape(), Shape::new(shape));
    }
}

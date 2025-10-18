#[burn_tensor_testgen::testgen(tri_mask_batch)]
mod tests {
    use super::*;
    use burn_tensor::Shape;

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

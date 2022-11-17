#[burn_tensor_testgen::testgen(mask)]
mod tests {
    use super::*;
    use burn_tensor::{BoolTensor, Data, Tensor};

    #[test]
    fn should_support_mask_ops() {
        let tensor = Tensor::<TestBackend, 2>::from_data(Data::from([[1.0, 7.0], [2.0, 3.0]]));
        let mask =
            BoolTensor::<TestBackend, 2>::from_data(Data::from([[true, false], [false, true]]));

        let data_actual = tensor.mask_fill(&mask, 2.0).to_data();

        let data_expected = Data::from([[2.0, 7.0], [2.0, 2.0]]);
        assert_eq!(data_expected, data_actual);
    }
}

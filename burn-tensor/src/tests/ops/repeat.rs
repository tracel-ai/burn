#[burn_tensor_testgen::testgen(repeat)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_repeat_ops() {
        let data = Data::from([[0.0, 1.0, 2.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.repeat(0, 4).into_data();

        let data_expected = Data::from([
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]);
        assert_eq!(data_expected, data_actual);
    }
}

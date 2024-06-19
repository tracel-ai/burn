#[burn_tensor_testgen::testgen(neg)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Tensor, TensorData};

    #[test]
    fn should_support_neg_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.neg();
        let expected = TensorData::from([[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]])
            .convert::<<TestBackend as Backend>::FloatElem>();

        output.into_data().assert_eq(&expected, true);
    }
}

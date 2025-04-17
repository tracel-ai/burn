#[burn_tensor_testgen::testgen(q_sinh)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_support_sinh_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.sinh();
        let expected = TensorData::from([[0.0000, 1.1752, 3.6269], [10.0179, 27.2899, 74.2032]]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::absolute(1e-1));
    }
}

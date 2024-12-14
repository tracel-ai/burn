#[burn_tensor_testgen::testgen(q_neg)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_neg_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.neg();
        let expected = TensorData::from([[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]).convert::<f32>();

        // -0.0 is represented differently than 0.0 so we make sure the values are the same in f32
        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}

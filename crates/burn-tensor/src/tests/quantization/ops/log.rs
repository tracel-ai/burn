#[burn_tensor_testgen::testgen(q_log)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_support_log_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.log();
        let expected = TensorData::from([
            [-f32::INFINITY, 0.0, core::f32::consts::LN_2],
            [1.0986, 1.3862, 1.6094],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}

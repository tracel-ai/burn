#[burn_tensor_testgen::testgen(q_floor)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_support_floor_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8([
            [24.0423, 87.9478, 76.1838],
            [59.6929, 43.8169, 94.8826],
        ]);

        let output = tensor.floor();
        let expected = TensorData::from([[24., 87., 76.], [59., 43., 95.]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}

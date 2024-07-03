#[burn_tensor_testgen::testgen(tanh)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_tanh_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.tanh();
        let expected = TensorData::from([[0.0, 0.7615, 0.9640], [0.9950, 0.9993, 0.9999]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}

#[burn_tensor_testgen::testgen(sin)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_sin_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.sin();
        let expected = TensorData::from([[0.0, 0.8414, 0.9092], [0.1411, -0.7568, -0.9589]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}

#[burn_tensor_testgen::testgen(ad_softmax)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, activation};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_softmax_grad() {
        let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);
        let device = Default::default();
        let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(data_1, &device).require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = activation::softmax(tensor_3, 1).matmul(tensor_2.clone());

        let grads = tensor_4.backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[1.179665, 1.179661], [0.005462, 0.005463]]);

        let tolerance = Tolerance::rel_abs(1e-5, 1e-4)
            // Softmax in f16 is not as accurate. For more accurate results, users probably want to upcast the input to f32.
            .set_half_precision_relative(5e-2)
            .set_half_precision_absolute(5.5e-2);
        grad_1
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        let expected = TensorData::from([[0.253469, 0.286237], [0.528630, 2.931664]]);
        grad_2
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);
    }

    #[test]
    fn test_log_softmax_grad() {
        let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);
        let device = Default::default();
        let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(data_1, &device).require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = activation::log_softmax(tensor_3, 1).matmul(tensor_2.clone());

        let grads = tensor_4.backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[-4.3939, -4.3939], [-12.9709, -12.9709]]);
        let tolerance = Tolerance::rel_abs(1e-4, 1e-5).set_half_precision_relative(5e-3);
        grad_1
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        let expected = TensorData::from([[30.5984, -47.2267], [55.9631, -56.5914]]);
        grad_2
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);
    }

    #[test]
    fn test_quiet_softmax_grad() {
        let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

        let device = Default::default();
        let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(data_1, &device).require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = activation::softmax(tensor_3, 1).matmul(tensor_2.clone());

        let grads = tensor_4.backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[1.179665, 1.179661], [0.005462, 0.005463]]);

        let tolerance = Tolerance::rel_abs(1e-5, 1e-4)
            // Softmax in f16 is not as accurate. For more accurate results, users probably want to upcast the input to f32.
            .set_half_precision_relative(5e-2)
            .set_half_precision_absolute(5.5e-2);
        grad_1
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        let expected = TensorData::from([[0.253469, 0.286237], [0.528630, 2.931664]]);
        grad_2
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);
    }
}

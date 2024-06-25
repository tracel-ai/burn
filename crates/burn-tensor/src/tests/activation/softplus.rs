#[burn_tensor_testgen::testgen(softplus)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_softplus_d2() {
        let tensor = Tensor::<TestBackend, 2>::from([
            [-0.4240, -0.9574, -0.2215],
            [-0.5767, 0.7218, -0.1620],
        ]);

        let output = activation::softplus(tensor.clone(), 1.0);
        let expected = TensorData::from([[0.5034, 0.3249, 0.5885], [0.4458, 1.1178, 0.6154]]);

        output.into_data().assert_approx_eq(&expected, 4);

        let output = activation::softplus(tensor, 2.0);
        let expected = TensorData::from([[0.1782, 0.0687, 0.2480], [0.1371, 0.8277, 0.2721]]);

        output.into_data().assert_approx_eq(&expected, 4);
    }
}

#[burn_tensor_testgen::testgen(mish)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_mish() {
        let tensor =
            TestTensor::<2>::from([[-0.4240, -0.9574, -0.2215], [-0.5767, 0.7218, -0.1620]]);

        let output = activation::mish(tensor);
        let expected = TensorData::from([[-0.1971, -0.3006, -0.1172], [-0.2413, 0.5823, -0.0888]]);

        output.into_data().assert_approx_eq(&expected, 4);
    }
}

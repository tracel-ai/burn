#[burn_tensor_testgen::testgen(leaky_relu)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_leaky_relu_d2() {
        let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, -4.0, 5.0]]);

        let output = activation::leaky_relu(tensor, 0.01);

        output.into_data().assert_eq(
            &TensorData::from([[0.0, -0.01, 2.0], [3.0, -0.04, 5.0]]),
            false,
        );
    }
}

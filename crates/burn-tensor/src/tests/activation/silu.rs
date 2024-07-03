#[burn_tensor_testgen::testgen(silu)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_silu() {
        let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

        let output = activation::silu(tensor);
        let expected = TensorData::from([[0.7311, 1.7616], [2.8577, 3.9281]]);

        output.into_data().assert_approx_eq(&expected, 4);
    }
}

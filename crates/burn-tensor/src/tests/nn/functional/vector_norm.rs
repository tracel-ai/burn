#[burn_tensor_testgen::testgen(nn_fn_vector_norm)]
mod tests {
    use super::*;
    use burn_tensor::nn::functional::linear;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_linear_1d() {
        let weight = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

        let x = TestTensor::<1>::from([1.0, 2.0]);

        linear(x.clone(), weight.clone(), None)
            .into_data()
            .assert_eq(
                &TensorData::from([7.0, 10.0]).convert_dtype(x.dtype()),
                true,
            );
    }

    #[test]
    fn test_linear_forward_no_bias() {
        let weight = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

        let x = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]]);

        linear(x.clone(), weight.clone(), None)
            .into_data()
            .assert_eq(
                &TensorData::from([[[7.0, 10.0], [15.0, 22.0]], [[-7.0, -10.0], [-15.0, -22.0]]])
                    .convert_dtype(x.dtype()),
                true,
            );
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let weight = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let bias = Some(TestTensor::<1>::from([1.0, -1.0]));

        let x = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]]);

        linear(x.clone(), weight.clone(), bias.clone())
            .into_data()
            .assert_eq(
                &TensorData::from([[[8.0, 9.0], [16.0, 21.0]], [[-6.0, -11.0], [-14.0, -23.0]]])
                    .convert_dtype(x.dtype()),
                true,
            );
    }
}

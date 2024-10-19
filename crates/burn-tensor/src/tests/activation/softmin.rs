#[burn_tensor_testgen::testgen(softmin)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_softmin_d2() {
        let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

        let output = activation::softmin(tensor, 1);
        let expected = TensorData::from([[9.9753e-01, 2.4726e-03], [1.1254e-07, 1.0000e+00]]);

        output.into_data().assert_approx_eq(&expected, 4);
    }
}

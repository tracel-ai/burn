#[burn_tensor_testgen::testgen(tan)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_tan_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.tan();
        let expected = TensorData::from([[0.0, 1.5574, -2.1850], [-0.1425, 1.1578, -3.3805]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}

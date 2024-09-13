#[burn_tensor_testgen::testgen(cos)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_cos_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.cos();
        let expected = TensorData::from([[1.0, 0.5403, -0.4161], [-0.9899, -0.6536, 0.2836]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}

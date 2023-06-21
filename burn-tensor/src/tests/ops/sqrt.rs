#[burn_tensor_testgen::testgen(sqrt)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};
    use core::f32::consts::SQRT_2;

    #[test]
    fn should_support_sqrt_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.sqrt().into_data();

        let data_expected = Data::from([[0.0, 1.0, SQRT_2], [1.73205, 2.0, 2.2360]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}

use super::super::TestBackend;
use burn_tensor::{Data, Tensor};

#[test]
fn should_support_erf_ops() {
    let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = Tensor::<TestBackend, 2>::from_data(data);

    let data_actual = tensor.erf().into_data();

    let data_expected = Data::from([[0.0000, 0.8427, 0.9953], [1.0000, 1.0000, 1.0000]]);
    data_expected.assert_approx_eq(&data_actual, 3);
}

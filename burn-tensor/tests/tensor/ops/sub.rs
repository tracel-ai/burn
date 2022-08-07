use super::super::TestBackend;
use burn_tensor::{Data, Tensor};

#[test]
fn should_support_sub_ops() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data_2 = Data::from([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);
    let data_expected = Data::from([[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);
    let tensor_2 = Tensor::<2, TestBackend>::from_data(data_2);

    let data_actual = (tensor_1 - tensor_2).into_data();

    assert_eq!(data_expected, data_actual);
}

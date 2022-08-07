use super::super::TestBackend;
use burn_tensor::{Data, Tensor};

#[test]
fn should_support_full_indexing_1d() {
    let data = Data::from([0.0, 1.0, 2.0]);
    let tensor = Tensor::<1, TestBackend>::from_data(data.clone());

    let data_actual = tensor.index([0..3]).into_data();

    assert_eq!(data, data_actual);
}

#[test]
fn should_support_partial_indexing_1d() {
    let data = Data::from([0.0, 1.0, 2.0]);
    let tensor = Tensor::<1, TestBackend>::from_data(data.clone());

    let data_actual = tensor.index([1..3]).into_data();

    let data_expected = Data::from([1.0, 2.0]);
    assert_eq!(data_expected, data_actual);
}

#[test]
fn should_support_full_indexing_2d() {
    let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = Tensor::<2, TestBackend>::from_data(data.clone());

    let data_actual_1 = tensor.index([0..2]).into_data();
    let data_actual_2 = tensor.index([0..2, 0..3]).into_data();

    assert_eq!(data, data_actual_1);
    assert_eq!(data, data_actual_2);
}

#[test]
fn should_support_partial_indexing_2d() {
    let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = Tensor::<2, TestBackend>::from_data(data.clone());

    let data_actual = tensor.index([0..2, 0..2]).into_data();

    let data_expected = Data::from([[0.0, 1.0], [3.0, 4.0]]);
    assert_eq!(data_expected, data_actual);
}

#[test]
fn should_support_indexe_assign_1d() {
    let data = Data::from([0.0, 1.0, 2.0]);
    let data_assigned = Data::from([10.0, 5.0]);

    let tensor = Tensor::<1, TestBackend>::from_data(data.clone());
    let tensor_assigned = Tensor::<1, TestBackend>::from_data(data_assigned.clone());

    let data_actual = tensor.index_assign([0..2], &tensor_assigned).into_data();

    let data_expected = Data::from([10.0, 5.0, 2.0]);
    assert_eq!(data_expected, data_actual);
}

#[test]
fn should_support_indexe_assign_2d() {
    let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data_assigned = Data::from([[10.0, 5.0]]);

    let tensor = Tensor::<2, TestBackend>::from_data(data.clone());
    let tensor_assigned = Tensor::<2, TestBackend>::from_data(data_assigned.clone());

    let data_actual = tensor
        .index_assign([1..2, 0..2], &tensor_assigned)
        .into_data();

    let data_expected = Data::from([[0.0, 1.0, 2.0], [10.0, 5.0, 5.0]]);
    assert_eq!(data_expected, data_actual);
}

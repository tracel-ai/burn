use super::*;
use burn_tensor::IndexingUpdateOp;

#[test]
fn shape_operations_support_negative_dims() {
    let device = Default::default();
    let tensor = TestTensor::<4>::ones([2, 1, 3, 1], &device);

    let expected: TestTensor<3> = tensor.clone().squeeze_dim(1);
    let actual: TestTensor<3> = tensor.clone().squeeze_dim(-3_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected: TestTensor<2> = tensor.clone().squeeze_dims(&[1, 3]);
    let actual: TestTensor<2> = tensor.squeeze_dims(&[-3_i64, -1]);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    let expected: TestTensor<3> = tensor.clone().unsqueeze_dim(2);
    let actual: TestTensor<3> = tensor.clone().unsqueeze_dim(-1_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected: TestTensor<3> = tensor.clone().unsqueeze_dim(0);
    let actual: TestTensor<3> = tensor.unsqueeze_dim(-3_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);

    // The insertion domain is based on the input rank even when the requested
    // output rank adds more than one singleton dimension.
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let expected: TestTensor<4> = tensor.clone().unsqueeze_dim(2);
    let actual: TestTensor<4> = tensor.clone().unsqueeze_dim(-1_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected: TestTensor<4> = tensor.clone().unsqueeze_dim(1);
    let actual: TestTensor<4> = tensor.unsqueeze_dim(-2_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn flip_and_movedim_support_all_index_types() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..24, &device)
        .reshape([2, 3, 4])
        .float();

    let expected = tensor.clone().flip([0, 2]);
    let actual = tensor.clone().flip([-3_i64, -1]);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected = tensor.clone().movedim(0, 2);
    let actual = tensor.clone().movedim(0_u8, -1_i16);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected = tensor.clone().movedim(vec![0, 1], vec![1, 0]);
    let actual = tensor.movedim(vec![-3_i64, -2], vec![-2_i64, -3]);
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn indexing_operations_support_negative_dims() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);

    let expected = tensor.clone().slice_dim(1, 1..);
    let actual = tensor.clone().slice_dim(-1_i64, 1..);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let indices = TestTensorInt::from_ints([[2, 1, 0], [0, 2, 1]], &device);
    let expected = tensor.clone().gather(1, indices.clone());
    let actual = tensor.clone().gather(-1_i64, indices);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let values = TestTensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let indices = TestTensorInt::from_ints([[1, 0, 2], [1, 2, 0]], &device);
    let base = TestTensor::<2>::zeros([2, 3], &device);
    let expected = base
        .clone()
        .scatter(1, indices.clone(), values.clone(), IndexingUpdateOp::Add);
    let actual = base.scatter(-1_i64, indices, values, IndexingUpdateOp::Add);
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn concatenation_and_repeat_support_negative_dims() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    let expected = tensor.clone().repeat_dim(1, 2);
    let actual = tensor.clone().repeat_dim(-1_i64, 2);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let other = TestTensor::<2>::from_data([[7.0], [8.0]], &device);
    let expected = TestTensor::cat(vec![tensor.clone(), other.clone()], 1);
    let actual = TestTensor::cat(vec![tensor.clone(), other], -1_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected: TestTensor<3> = TestTensor::stack(vec![tensor.clone(), tensor.clone()], 2);
    let actual: TestTensor<3> = TestTensor::stack(vec![tensor.clone(), tensor.clone()], -1_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected: TestTensor<3> = TestTensor::stack(vec![tensor.clone(), tensor.clone()], 0);
    let actual: TestTensor<3> = TestTensor::stack(vec![tensor.clone(), tensor.clone()], -3_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected: TestTensor<4> = TestTensor::stack(vec![tensor.clone(), tensor.clone()], 2);
    let actual: TestTensor<4> = TestTensor::stack(vec![tensor.clone(), tensor.clone()], -1_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected: TestTensor<4> = TestTensor::stack(vec![tensor.clone(), tensor.clone()], 1);
    let actual: TestTensor<4> = TestTensor::stack(vec![tensor.clone(), tensor], -2_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn partition_operations_support_negative_dims() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);

    let expected = tensor.clone().narrow(1, 1, 2);
    let actual = tensor.clone().narrow(-1_i64, 1, 2);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected = tensor.clone().chunk(2, 1);
    let actual = tensor.clone().chunk(2, -1_i64);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        actual.into_data().assert_eq(&expected.into_data(), false);
    }

    let expected = tensor.clone().split(2, 1);
    let actual = tensor.clone().split(2, -1_i64);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        actual.into_data().assert_eq(&expected.into_data(), false);
    }

    let expected = tensor.clone().split_with_sizes(vec![1, 2], 1);
    let actual = tensor.split_with_sizes(vec![1, 2], -1_i64);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        actual.into_data().assert_eq(&expected.into_data(), false);
    }
}

#[test]
fn iteration_and_boolean_reductions_support_negative_dims() {
    let tensor = TestTensor::<2>::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);

    let expected = tensor.clone().iter_dim(1).collect::<Vec<_>>();
    let actual = tensor.clone().iter_dim(-1_i64).collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        actual.into_data().assert_eq(&expected.into_data(), false);
    }

    let expected = tensor.clone().any_dim(1);
    let actual = tensor.clone().any_dim(-1_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);

    let expected = tensor.clone().all_dim(1);
    let actual = tensor.all_dim(-1_i64);
    actual.into_data().assert_eq(&expected.into_data(), false);
}

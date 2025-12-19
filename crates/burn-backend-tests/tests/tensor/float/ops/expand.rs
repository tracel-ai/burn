use super::*;
use burn_tensor::TensorData;

#[test]
fn expand_2d() {
    let tensor = TestTensor::<1>::from_floats([1.0, 2.0, 3.0], &Default::default());
    let output = tensor.expand([3, 3]);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        false,
    );

    let tensor = TestTensor::<1>::from_floats([4.0, 7.0, 2.0, 3.0], &Default::default());
    let output = tensor.expand([2, 4]);

    output.into_data().assert_eq(
        &TensorData::from([[4.0, 7.0, 2.0, 3.0], [4.0, 7.0, 2.0, 3.0]]),
        false,
    );
}

#[test]
fn expand_3d() {
    let tensor = TestTensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &Default::default());
    let output = tensor.expand([3, 2, 2]);
    let expected = TensorData::from([
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn expand_higher_dimensions() {
    let tensor = TestTensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &Default::default());
    let output = tensor.expand([2, 3, 4]);
    let expected = TensorData::from([
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn expand_sum_3d() {
    let tensor = TestTensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &Default::default());
    let output = tensor.expand([3, 2, 2]).sum_dim(0);
    let expected = TensorData::from([[[3.0, 6.0], [9.0, 12.0]]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn broadcast_single() {
    let tensor = TestTensor::<1>::from_floats([1.0], &Default::default());
    let output = tensor.expand([2, 3]);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), false);
}

#[test]
#[should_panic]
fn should_fail_expand_incompatible_shapes() {
    let tensor = TestTensor::<1>::from_floats([1.0, 2.0, 3.0], &Default::default());
    let _expanded_tensor = tensor.expand([2, 2]);
}

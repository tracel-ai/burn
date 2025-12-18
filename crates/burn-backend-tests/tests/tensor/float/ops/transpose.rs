use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_transpose_ops() {
    let tensor = TestTensor::<3>::from_floats(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    // Check the .t() alias.
    let output = tensor.t();

    let expected = TensorData::from([
        [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
        [[6.0, 9.0], [7.0, 10.0], [8.0, 11.0]],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_transpose_maybe_fused_with_one() {
    let tensor = TestTensor::<3>::from_floats(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );
    let ones = TestTensor::<3>::ones([1, 1, 1], &Default::default());

    let output = tensor.transpose();
    let expected = TensorData::from([
        [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
        [[6.0, 9.0], [7.0, 10.0], [8.0, 11.0]],
    ]);
    let expected_ones = TensorData::from([[[1.0]]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
    ones.into_data()
        .assert_approx_eq::<FloatElem>(&expected_ones, Tolerance::default());
}

#[test]
fn should_support_swap_dims_no_op() {
    let tensor = TestTensor::<3>::from_floats(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let output = tensor.swap_dims(0, 0);
    let expected = TensorData::from([
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_swap_dims() {
    let tensor = TestTensor::<3>::from_floats(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let output = tensor.swap_dims(0, 2);
    let expected = TensorData::from([
        [[0.0, 6.0], [3.0, 9.0]],
        [[1.0, 7.0], [4.0, 10.0]],
        [[2.0, 8.0], [5.0, 11.0]],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_swap_dims_neg_index() {
    let tensor = TestTensor::<3>::from_floats(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let output = tensor.swap_dims(-3, -1);
    let expected = TensorData::from([
        [[0.0, 6.0], [3.0, 9.0]],
        [[1.0, 7.0], [4.0, 10.0]],
        [[2.0, 8.0], [5.0, 11.0]],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

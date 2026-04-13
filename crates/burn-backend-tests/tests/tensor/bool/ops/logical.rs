use super::*;
use burn_tensor::TensorData;

#[test]
fn test_bool_and() {
    let tensor1 = TestTensorBool::<2>::from([[false, true, false], [true, false, true]]);
    let tensor2 = TestTensorBool::<2>::from([[true, true, false], [false, false, true]]);
    let data_actual = tensor1.bool_and(tensor2).into_data();
    let data_expected = TensorData::from([[false, true, false], [false, false, true]]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_bool_or() {
    let tensor1 = TestTensorBool::<2>::from([[false, true, false], [true, false, true]]);
    let tensor2 = TestTensorBool::<2>::from([[true, true, false], [false, false, true]]);
    let data_actual = tensor1.bool_or(tensor2).into_data();
    let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_bool_xor() {
    let tensor1 = TestTensorBool::<2>::from([[false, true, false], [true, false, true]]);
    let tensor2 = TestTensorBool::<2>::from([[true, true, false], [false, false, true]]);
    let data_actual = tensor1.bool_xor(tensor2).into_data();
    let data_expected = TensorData::from([[true, false, false], [true, false, false]]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_bool_or_vec() {
    let device = Default::default();
    let tensor1 = TestTensorBool::<1>::full([256], 0, &device);
    let tensor2 = TestTensorBool::<1>::full([256], 1, &device);
    let data_actual = tensor1.bool_or(tensor2).into_data();
    let data_expected = TensorData::from([true; 256]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_bool_and_vec() {
    let device = Default::default();
    let tensor1 = TestTensorBool::<1>::full([256], 0, &device);
    let tensor2 = TestTensorBool::<1>::full([256], 1, &device);
    let data_actual = tensor1.bool_and(tensor2).into_data();
    let data_expected = TensorData::from([false; 256]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_bool_and_broadcast_scalar_lhs() {
    // [1, 1] AND [2, 3] broadcasts to [2, 3].
    let device = Default::default();
    let tensor1 = TestTensorBool::<2>::from_data(TensorData::from([[true]]), &device);
    let tensor2 = TestTensorBool::<2>::from_data(
        TensorData::from([[true, false, true], [false, true, false]]),
        &device,
    );

    let actual = tensor1.bool_and(tensor2).into_data();
    let expected = TensorData::from([[true, false, true], [false, true, false]]);
    expected.assert_eq(&actual, false);
}

#[test]
fn test_bool_or_broadcast_scalar_lhs() {
    // [1, 1]=false OR [2, 3] broadcasts to [2, 3] and equals rhs.
    let device = Default::default();
    let tensor1 = TestTensorBool::<2>::from_data(TensorData::from([[false]]), &device);
    let tensor2 = TestTensorBool::<2>::from_data(
        TensorData::from([[true, false, true], [false, true, false]]),
        &device,
    );

    let actual = tensor1.bool_or(tensor2).into_data();
    let expected = TensorData::from([[true, false, true], [false, true, false]]);
    expected.assert_eq(&actual, false);
}

#[test]
fn test_bool_and_broadcast_rank_mismatch() {
    // [2, 3, 4] AND [1, 3, 4] (unsqueezed from [3, 4]) broadcasts along dim 0.
    let device = Default::default();
    let lhs = TestTensorBool::<3>::from_data(
        TensorData::from([
            [
                [true, false, false, false],
                [true, false, true, false],
                [false, true, true, true],
            ],
            [
                [true, false, false, true],
                [false, false, true, true],
                [true, true, false, true],
            ],
        ]),
        &device,
    );
    let rhs = TestTensorBool::<2>::from_data(
        TensorData::from([
            [false, false, true, true],
            [false, true, true, false],
            [false, false, false, true],
        ]),
        &device,
    );

    let actual = lhs.bool_and(rhs.unsqueeze::<3>()).into_data();
    let expected = TensorData::from([
        [
            [false, false, false, false],
            [false, false, true, false],
            [false, false, false, true],
        ],
        [
            [false, false, false, true],
            [false, false, true, false],
            [false, false, false, true],
        ],
    ]);
    expected.assert_eq(&actual, false);
}

#[test]
fn test_bool_or_broadcast_rank_mismatch() {
    // [2, 3, 4] OR [1, 3, 4] (unsqueezed from [3, 4]) broadcasts along dim 0.
    let device = Default::default();
    let lhs = TestTensorBool::<3>::from_data(
        TensorData::from([
            [
                [true, false, false, false],
                [true, false, true, false],
                [false, true, true, true],
            ],
            [
                [true, false, false, true],
                [false, false, true, true],
                [true, true, false, true],
            ],
        ]),
        &device,
    );
    let rhs = TestTensorBool::<2>::from_data(
        TensorData::from([
            [false, false, true, true],
            [false, true, true, false],
            [false, false, false, true],
        ]),
        &device,
    );

    let actual = lhs.bool_or(rhs.unsqueeze::<3>()).into_data();
    let expected = TensorData::from([
        [
            [true, false, true, true],
            [true, true, true, false],
            [false, true, true, true],
        ],
        [
            [true, false, true, true],
            [false, true, true, true],
            [true, true, false, true],
        ],
    ]);
    expected.assert_eq(&actual, false);
}

#[test]
fn test_bool_xor_broadcast_rank_mismatch() {
    // [2, 2, 2] XOR [1, 2, 2] broadcasts along the leading dim.
    let device = Default::default();
    let lhs = TestTensorBool::<3>::from_data(
        TensorData::from([
            [[true, false], [false, true]],
            [[true, true], [false, false]],
        ]),
        &device,
    );
    let rhs =
        TestTensorBool::<2>::from_data(TensorData::from([[true, false], [false, true]]), &device);

    let actual = lhs.bool_xor(rhs.unsqueeze::<3>()).into_data();
    let expected = TensorData::from([
        [[false, false], [false, false]],
        [[false, true], [false, true]],
    ]);
    expected.assert_eq(&actual, false);
}

#[test]
fn test_bool_and_broadcast_row_lhs() {
    // [1, 3] AND [2, 3] broadcasts lhs along dim 0 only. Distinct from
    // the scalar_lhs pattern, which broadcasts along both dims.
    let device = Default::default();
    let lhs = TestTensorBool::<2>::from_data(TensorData::from([[true, false, true]]), &device);
    let rhs = TestTensorBool::<2>::from_data(
        TensorData::from([[true, true, false], [false, true, true]]),
        &device,
    );

    let actual = lhs.bool_and(rhs).into_data();
    let expected = TensorData::from([[true, false, false], [false, false, true]]);
    expected.assert_eq(&actual, false);
}

#[test]
fn test_bool_and_broadcast_mutual() {
    // [3, 1] AND [1, 3] - both operands need expansion along different
    // axes, producing [3, 3]. Exercises the "neither operand matches the
    // output shape" path through broadcast_binary.
    let device = Default::default();
    let lhs = TestTensorBool::<2>::from_data(TensorData::from([[true], [false], [true]]), &device);
    let rhs = TestTensorBool::<2>::from_data(TensorData::from([[true, false, true]]), &device);

    let actual = lhs.bool_and(rhs).into_data();
    let expected = TensorData::from([
        [true, false, true],
        [false, false, false],
        [true, false, true],
    ]);
    expected.assert_eq(&actual, false);
}

#[test]
fn test_bool_and_broadcast_4d() {
    // 4D broadcast: [2, 1, 2, 1] AND [1, 2, 1, 2] -> [2, 2, 2, 2].
    // Confirms the fix is rank-agnostic. Pre-fix, both operands had 4
    // storage elements so the inplace helper ran without panicking, but
    // the output silently kept lhs's [2, 1, 2, 1] shape - shape assert
    // catches that.
    let device = Default::default();
    let lhs = TestTensorBool::<4>::from_data(
        TensorData::from([[[[true], [false]]], [[[false], [true]]]]),
        &device,
    );
    let rhs = TestTensorBool::<4>::from_data(
        TensorData::from([[[[true, false]], [[false, true]]]]),
        &device,
    );

    let actual = lhs.bool_and(rhs).into_data();
    let expected = TensorData::from([
        [
            [[true, false], [false, false]],
            [[false, true], [false, false]],
        ],
        [
            [[false, false], [true, false]],
            [[false, false], [false, true]],
        ],
    ]);
    expected.assert_eq(&actual, false);
}

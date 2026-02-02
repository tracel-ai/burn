use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_reshape_maybe_fused_1() {
    let tensor = TestTensorInt::arange(0..32, &Default::default());
    let tensor0 = TestTensorInt::zeros([8, 4, 8], &Default::default());
    let tensor1 = tensor.clone().reshape([1, 4, 8]);
    let output = tensor0 + tensor1;

    let expected = TensorData::from([
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ],
    ]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_reshape_maybe_fused_2() {
    let tensor = TestTensorInt::<3>::from_data([[[0, 2], [1, 2]]], &Default::default());
    let tensor1 = tensor.reshape([2, 2, 1]);
    let tensor2 = TestTensorInt::<3>::full([2, 2, 4], 4, &Default::default());
    let output = tensor2 + tensor1;

    let expected_tensor1 =
        TensorData::from([[[4, 4, 4, 4], [6, 6, 6, 6]], [[5, 5, 5, 5], [6, 6, 6, 6]]]);
    output.into_data().assert_eq(&expected_tensor1, false);
}

#[test]
fn should_support_reshape_maybe_fused_3() {
    let tensor = TestTensorInt::<3>::from_data([[[0, 2], [1, 2]]], &Default::default());
    let tensor1 = tensor.reshape([2, 2, 1]);
    let _tensor2 = TestTensorInt::<3>::full([2, 2, 3], 5, &Default::default());

    let expected_tensor1 = TensorData::from([[[0], [2]], [[1], [2]]]);
    tensor1.into_data().assert_eq(&expected_tensor1, false);
}

#[test]
fn should_support_reshape_maybe_fused_4() {
    let tensor = TestTensorInt::<3>::from_data([[[0, 2], [1, 2]]], &Default::default());
    let tensor2 = TestTensorInt::<3>::full([2, 2, 4], 4, &Default::default());
    let tensor2 = tensor2.swap_dims(0, 1);
    let tensor1 = tensor.reshape([2, 2, 1]);
    let output = tensor2 + tensor1;

    let expected_tensor1 =
        TensorData::from([[[4, 4, 4, 4], [6, 6, 6, 6]], [[5, 5, 5, 5], [6, 6, 6, 6]]]);
    output.into_data().assert_eq(&expected_tensor1, false);
}

#[test]
fn should_support_reshape_maybe_fused_5() {
    let tensor = TestTensorInt::<3>::from_data([[[0], [1], [2], [3]]], &Default::default());
    let tensor1 = tensor.clone().reshape([2, 1, 2]);
    let tensor2 = TestTensorInt::<3>::full([2, 4, 2], 0, &Default::default());
    let output = tensor2.clone() + tensor1 + tensor.clone();

    let expected_tensor1 = TensorData::from([
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        [[2, 3], [3, 4], [4, 5], [5, 6]],
    ]);
    output.into_data().assert_eq(&expected_tensor1, false);
}

#[test]
fn should_support_reshape_maybe_fused_6() {
    let device = Default::default();

    let tensor1 = TestTensorInt::arange(0..32, &device);
    let tensor1 = tensor1.reshape([2, 4, 4]);

    let tensor2 = TestTensorInt::arange(0..16, &device);
    let tensor2 = tensor2.reshape([1, 4, 4]);

    let tensor3 = TestTensorInt::arange(0..8, &device);
    let tensor3 = tensor3.reshape([4, 1, 2]);
    let tensor3 = tensor3.swap_dims(0, 2);

    let out = tensor1 + tensor2 + tensor3;

    let expected = TensorData::from([
        [
            [0, 4, 8, 12],
            [8, 12, 16, 20],
            [16, 20, 24, 28],
            [24, 28, 32, 36],
        ],
        [
            [17, 21, 25, 29],
            [25, 29, 33, 37],
            [33, 37, 41, 45],
            [41, 45, 49, 53],
        ],
    ]);
    out.to_data().assert_eq(&expected, false);
}

// Skip on metal - cubecl autotune error
// Enable once this issue is fixed: https://github.com/tracel-ai/burn/issues/4327
#[cfg(not(feature = "metal"))]
#[test]
fn should_support_multiple_reshapes_cloned_tensor() {
    let device = Default::default();

    let lhs = TestTensorInt::<1>::arange(0..4, &device).reshape([2, 2]);
    // fusion should preserve correct strides when operating on the same tensor
    let rhs = lhs.clone();

    let lhs = lhs.reshape([2, 2, 1]);
    let rhs = rhs.reshape([1, 2, 2]);

    let p = lhs.mul(rhs);

    let s = p.sum_dim(1);

    let out = s.reshape([2, 2]);

    out.into_data()
        .assert_eq(&TensorData::from([[2, 3], [6, 11]]), false);
}

#[test]
fn should_support_reshape_int() {
    let data = TensorData::from([0, 1, 2]);
    let tensor = TestTensorInt::<1>::from_data(data, &Default::default());

    let output = tensor.clone().reshape([1, 3]);
    let expected = TensorData::from([[0, 1, 2]]);

    output.into_data().assert_eq(&expected, false);
}

use super::*;
use burn_tensor::TensorData;

#[test]
fn test_int_matmul_d2() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_ints([[1, 7], [2, 3], [1, 5]], &device);
    let tensor_2 = TestTensorInt::<2>::from_ints([[4, 7, 5], [2, 3, 5]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[18, 28, 40], [14, 23, 25], [14, 22, 30]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_d3() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<3>::from_ints([[[1, 7], [2, 3]]], &device);
    let tensor_2 = TestTensorInt::<3>::from_ints([[[4, 7], [2, 3]]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[[18, 28], [14, 23]]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_broadcast_1() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<3>::from_ints([[[1, 7], [2, 3]]], &device);
    let tensor_2 = TestTensorInt::from_ints([[[4, 7], [2, 3]], [[2, 5], [6, 3]]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[[18, 28], [14, 23]], [[44, 26], [22, 19]]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_broadcast_4d() {
    let device = Default::default();
    // [2, 1, 2, 2]
    let tensor_1 = TestTensorInt::<4>::from_ints([[[[1, 7], [2, 3]]], [[[2, 5], [6, 3]]]], &device);
    // [1, 2, 2, 2]
    let tensor_2 = TestTensorInt::from_ints([[[[9, 8], [1, 4]], [[2, 7], [3, 5]]]], &device);

    // [2, 1, 2, 2] @ [1, 2, 2, 2] -> [2, 2, 2, 2]
    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [[[16, 36], [21, 28]], [[23, 42], [13, 29]]],
        [[[23, 36], [57, 60]], [[19, 39], [21, 57]]],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_simple_1() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_ints([[5, 14], [14, 25]], &device);
    let tensor_2 = TestTensorInt::from_ints([[3, 4, 5], [0, 1, 2]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[15, 34, 53], [42, 81, 120]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_4_3() {
    if (IntElem::MAX as u32) < 324 {
        return;
    }

    let device = Default::default();
    let tensor_1 =
        TestTensorInt::<2>::from_ints([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], &device);
    let tensor_2 =
        TestTensorInt::from_ints([[0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[56, 62, 68], [152, 174, 196], [248, 286, 324]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_trivial() {
    if (IntElem::MAX as u32) < 506 {
        return;
    }

    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..16, &device).reshape([4, 4]);

    let tensor_3 = tensor_1.clone().matmul(tensor_1);

    tensor_3.into_data().assert_eq(
        &TensorData::from([
            [56, 62, 68, 74],
            [152, 174, 196, 218],
            [248, 286, 324, 362],
            [344, 398, 452, 506],
        ]),
        false,
    );
}

#[test]
fn test_int_matmul_trivial_transposed() {
    if (IntElem::MAX as u32) < 734 {
        return;
    }

    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..16, &device).reshape([4, 4]);

    let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

    tensor_3.into_data().assert_eq(
        &TensorData::from([
            [14, 38, 62, 86],
            [38, 126, 214, 302],
            [62, 214, 366, 518],
            [86, 302, 518, 734],
        ]),
        false,
    );
}

#[test]
fn test_int_matmul_4_8() {
    if (IntElem::MAX as u32) < 6092 {
        return;
    }

    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..32, &device).reshape([4, 8]);

    let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

    tensor_3.into_data().assert_eq(
        &TensorData::from([
            [140, 364, 588, 812],
            [364, 1100, 1836, 2572],
            [588, 1836, 3084, 4332],
            [812, 2572, 4332, 6092],
        ]),
        false,
    );
}

#[test]
fn test_int_matmul_simple_2() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_ints([[1, 2, 3, 4]], &device);
    let tensor_2 = TestTensorInt::from_ints([[3], [4], [5], [6]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[50]]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_matmul_simple_3() {
    let device = Default::default();
    let tensor_1 =
        TestTensorInt::<2>::from_ints([[3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]], &device);
    let tensor_2 = TestTensorInt::from_ints([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [9, 18, 27, 36],
        [12, 24, 36, 48],
        [15, 30, 45, 60],
        [18, 36, 54, 72],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn int_should_panic_when_inner_dimensions_are_not_equal() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_ints([[3, 3], [4, 4], [5, 5], [6, 6]], &device);
    let tensor_2 = TestTensorInt::from_ints([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [9, 18, 27, 36],
        [12, 24, 36, 48],
        [15, 30, 45, 60],
        [18, 36, 54, 72],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

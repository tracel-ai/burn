use super::*;
use burn_tensor::TensorData;
use burn_tensor::{ElementConversion, Tolerance, backend::Backend};

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
fn test_float_matmul_d2() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_floats([[1.0, 7.0], [2.0, 3.0], [1.0, 5.0]], &device);
    let tensor_2 = TestTensor::from_floats([[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[18.0, 28.0, 40.0], [14.0, 23.0, 25.0], [14.0, 22.0, 30.0]]);

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
fn test_float_matmul_d3() {
    let device = Default::default();
    let tensor_1 = TestTensor::<3>::from_floats([[[1.0, 7.0], [2.0, 3.0]]], &device);
    let tensor_2 = TestTensor::from_floats([[[4.0, 7.0], [2.0, 3.0]]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[[18.0, 28.0], [14.0, 23.0]]]);

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
fn test_float_matmul_broadcast_1() {
    let device = Default::default();
    let tensor_1 = TestTensor::<3>::from_floats([[[1.0, 7.0], [2.0, 3.0]]], &device);
    let tensor_2 = TestTensor::from_floats(
        [[[4.0, 7.0], [2.0, 3.0]], [[2.0, 5.0], [6.0, 3.0]]],
        &device,
    );

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[[18.0, 28.0], [14.0, 23.0]], [[44.0, 26.0], [22.0, 19.0]]]);

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
fn test_float_matmul_broadcast_4d() {
    let device = Default::default();
    // [2, 1, 2, 2]
    let tensor_1 = TestTensor::<4>::from_floats(
        [[[[1.0, 7.0], [2.0, 3.0]]], [[[2.0, 5.0], [6.0, 3.0]]]],
        &device,
    );
    // [1, 2, 2, 2]
    let tensor_2 = TestTensor::from_floats(
        [[[[9.0, 8.0], [1.0, 4.0]], [[2.0, 7.0], [3.0, 5.0]]]],
        &device,
    );

    // [2, 1, 2, 2] @ [1, 2, 2, 2] -> [2, 2, 2, 2]
    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [[[16.0, 36.0], [21.0, 28.0]], [[23.0, 42.0], [13.0, 29.0]]],
        [[[23.0, 36.0], [57.0, 60.0]], [[19.0, 39.0], [21.0, 57.0]]],
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
fn test_float_matmul_simple_1() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_floats([[5.0, 14.0], [14.0, 50.0]], &device);
    let tensor_2 = TestTensor::from_floats([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[15.0, 34.0, 53.0], [42.0, 106.0, 170.0]]);

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
fn test_float_matmul_4_3() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_floats(
        [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
        &device,
    );
    let tensor_2 = TestTensor::from_floats(
        [[0., 1., 2.], [4., 5., 6.], [8., 9., 10.], [12., 13., 14.]],
        &device,
    );

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[56., 62., 68.], [152., 174., 196.], [248., 286., 324.]]);

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
fn test_float_matmul_trivial() {
    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..16, &device)
        .reshape([4, 4])
        .float();

    let tensor_3 = tensor_1.clone().matmul(tensor_1);

    tensor_3.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([
            [56., 62., 68., 74.],
            [152., 174., 196., 218.],
            [248., 286., 324., 362.],
            [344., 398., 452., 506.],
        ]),
        Tolerance::default(),
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
fn test_float_matmul_trivial_transposed() {
    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..16, &device)
        .reshape([4, 4])
        .float();

    let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

    tensor_3.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([
            [14., 38., 62., 86.],
            [38., 126., 214., 302.],
            [62., 214., 366., 518.],
            [86., 302., 518., 734.],
        ]),
        Tolerance::default(),
    );
}

/// Regression test for batch bug in fused matmul
#[test]
fn test_float_matmul_vecmat_transposed_fused() {
    let device = Default::default();

    let batch1 = 1;
    let batch2 = 2;
    let batch = batch1 * batch2;
    let seq_length = 3;
    let d_model = 32;

    // Guard int arange limits
    #[allow(clippy::absurd_extreme_comparisons)]
    if IntElem::MAX < seq_length * d_model * batch {
        return;
    }
    if FloatElem::MAX.elem::<f64>() < 269493.0 {
        return;
    }

    let weight: TestTensor<4> = TestTensorInt::arange(0..d_model * batch, &device)
        .reshape([batch1, batch2, 1, d_model])
        .float();
    let signal: TestTensor<4> = TestTensorInt::arange(0..seq_length * d_model * batch, &device)
        .reshape([batch1, batch2, seq_length, d_model])
        .float();

    TestBackend::sync(&device).unwrap();
    let weight = weight.transpose();
    let out = signal.matmul(weight) + 5;
    let expected = TensorData::from([[
        [[10421.0], [26293.0], [42165.0]],
        [[172213.0], [220853.0], [269493.0]],
    ]]);
    expected.assert_approx_eq(&out.into_data(), Tolerance::<f32>::strict());
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
fn test_float_matmul_4_8() {
    let device = Default::default();

    let tensor_1 = TestTensorInt::<1>::arange(0..32, &device)
        .reshape([4, 8])
        .float();

    let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

    tensor_3.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([
            [140., 364., 588., 812.],
            [364., 1100., 1836., 2572.],
            [588., 1836., 3084., 4332.],
            [812., 2572., 4332., 6092.],
        ]),
        Tolerance::default(),
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
fn test_float_matmul_simple_2() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let tensor_2 = TestTensor::from_floats([[3.0], [4.0], [5.0], [6.0]], &device);

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([[50.0]]);

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
fn test_float_matmul_simple_3() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_floats(
        [[3., 3., 3.], [4., 4., 4.], [5., 5., 5.], [6., 6., 6.]],
        &device,
    );
    let tensor_2 = TestTensor::from_floats(
        [[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]],
        &device,
    );

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [9., 18., 27., 36.],
        [12., 24., 36., 48.],
        [15., 30., 45., 60.],
        [18., 36., 54., 72.],
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

#[test]
#[should_panic]
fn float_should_panic_when_inner_dimensions_are_not_equal() {
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_floats([[3., 3.], [4., 4.], [5., 5.], [6., 6.]], &device);
    let tensor_2 = TestTensor::from_floats(
        [[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]],
        &device,
    );

    let tensor_3 = tensor_1.matmul(tensor_2);
    let expected = TensorData::from([
        [9., 18., 27., 36.],
        [12., 24., 36., 48.],
        [15., 30., 45., 60.],
        [18., 36., 54., 72.],
    ]);

    tensor_3.into_data().assert_eq(&expected, false);
}

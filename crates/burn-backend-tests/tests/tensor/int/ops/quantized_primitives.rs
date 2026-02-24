use super::*;
use burn_tensor::{
    IntDType, TensorData,
    ops::{ConvOptions, FloatTensorOps, IntTensorOps},
};

#[test]
fn test_int_matmul_accum_u8_i32() {
    let device = Default::default();

    let lhs = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([[10u8, 12u8], [8u8, 9u8]]),
        &device,
    );
    let rhs = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([[2u8, 4u8], [6u8, 8u8]]),
        &device,
    );
    let lhs_zp = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([10u8]),
        &device,
    );
    let rhs_zp = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([5u8]),
        &device,
    );

    let output = <TestBackend as IntTensorOps<TestBackend>>::int_matmul_accum(
        lhs, lhs_zp, rhs, rhs_zp,
    );

    TestTensorInt::<2>::from_primitive(output)
        .into_data()
        .assert_eq(&TensorData::from([[2i32, 6i32], [5i32, -1i32]]), false);
}

#[test]
fn test_int_conv2d_accum_u8_i8_i32() {
    let device = Default::default();

    let x = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([[[[11u8, 12u8], [13u8, 14u8]]]]),
        &device,
    );
    let weight = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([[[[2i8]]]]),
        &device,
    );
    let x_zp = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([10u8]),
        &device,
    );
    let w_zp = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([1i8]),
        &device,
    );

    let output = <TestBackend as IntTensorOps<TestBackend>>::int_conv2d_accum(
        x,
        x_zp,
        weight,
        w_zp,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    );

    TestTensorInt::<4>::from_primitive(output)
        .into_data()
        .assert_eq(&TensorData::from([[[[1i32, 2i32], [3i32, 4i32]]]]), false);
}

#[test]
fn test_int_requantize_to_u8_and_i8() {
    let device = Default::default();

    let accum = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([[-3i32, -1i32, 0i32, 1i32, 3i32]]),
        &device,
    );
    let scale = <TestBackend as FloatTensorOps<TestBackend>>::float_from_data(
        TensorData::from([[1.25f32, 1.25, 1.25, 1.25, 1.25]]),
        &device,
    );

    let zp_u8 = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([128i32]),
        &device,
    );

    let output_u8 = <TestBackend as IntTensorOps<TestBackend>>::int_requantize(
        accum,
        scale,
        zp_u8,
        IntDType::U8,
    );

    TestTensorInt::<2>::from_primitive(output_u8)
        .into_data()
        .assert_eq(&TensorData::from([[124u8, 127u8, 128u8, 129u8, 132u8]]), false);

    let accum_i8 = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([[-200i32, -128i32, 0i32, 127i32, 200i32]]),
        &device,
    );
    let scale_1 = <TestBackend as FloatTensorOps<TestBackend>>::float_from_data(
        TensorData::from([[1.0f32, 1.0, 1.0, 1.0, 1.0]]),
        &device,
    );
    let zp_i8 = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([0i32]),
        &device,
    );

    let output_i8 = <TestBackend as IntTensorOps<TestBackend>>::int_requantize(
        accum_i8,
        scale_1,
        zp_i8,
        IntDType::I8,
    );

    TestTensorInt::<2>::from_primitive(output_i8)
        .into_data()
        .assert_eq(&TensorData::from([[-128i8, -128i8, 0i8, 127i8, 127i8]]), false);
}

#[test]
fn test_int_requantize_ties_to_even_and_saturation() {
    let device = Default::default();

    let accum = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([[-3i32, -1i32, 1i32, 3i32, 400i32, -400i32]]),
        &device,
    );
    let scale = <TestBackend as FloatTensorOps<TestBackend>>::float_from_data(
        TensorData::from([[0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5]]),
        &device,
    );
    let zero_point = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
        TensorData::from([0i32]),
        &device,
    );

    let output_i8 = <TestBackend as IntTensorOps<TestBackend>>::int_requantize(
        accum,
        scale,
        zero_point,
        IntDType::I8,
    );

    // [-1.5, -0.5, 0.5, 1.5, 200, -200] -> [-2, 0, 0, 2, 127, -128]
    TestTensorInt::<2>::from_primitive(output_i8)
        .into_data()
        .assert_eq(&TensorData::from([[-2i8, 0i8, 0i8, 2i8, 127i8, -128i8]]), false);
}

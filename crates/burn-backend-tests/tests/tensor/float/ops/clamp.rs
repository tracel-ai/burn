use super::*;
use burn_tensor::TensorData;

#[test]
fn clamp_min() {
    let device = Default::default();
    // test float tensor
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &device);

    let output = tensor.clamp_min(2.0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]]), false);

    // test int tensor
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let tensor = TestTensorInt::<2>::from_data(data, &device);
    let output = tensor.clamp_min(2);

    output
        .into_data()
        .assert_eq(&TensorData::from([[2, 2, 2], [3, 4, 5]]), false);
}

#[test]
fn clamp_max() {
    let device = Default::default();
    // test float tensor
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &device);

    let output = tensor.clamp_max(2.0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]]), false);

    // test int tensor
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let tensor = TestTensorInt::<2>::from_data(data, &device);
    let output = tensor.clamp_max(4);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 1, 2], [3, 4, 4]]), false);
}

#[test]
fn clamp_min_max() {
    let device = Default::default();
    // test float tensor
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &device);
    let output = tensor.clamp(1.0, 4.0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 2.0], [3.0, 4.0, 4.0]]), false);

    // test int tensor
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let tensor = TestTensorInt::<2>::from_data(data, &device);
    let output = tensor.clamp(1, 4);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 1, 2], [3, 4, 4]]), false);
}

#[test]
fn clamp_min_max_vec_should_compile() {
    let input = TestTensor::<2>::ones([2, 4], &Default::default());
    let output = input.clamp(0., 0.5);

    output.into_data().assert_eq(
        &TensorData::from([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
        false,
    );
}

// All burn backends should propagate NaN through clamp_min / clamp_max. See issue #5609.
#[test]
fn clamp_min_nan_propagation() {
    let tensor = TestTensor::<1>::from([f32::NAN, -1.0, 2.0]);

    let output = tensor.clamp_min(0.0).into_data().convert::<f32>();
    let values = output.as_slice::<f32>().unwrap();

    assert!(values[0].is_nan());
    assert_eq!(values[1..], [0.0, 2.0]);
}

#[test]
fn clamp_max_nan_propagation() {
    let tensor = TestTensor::<1>::from([f32::NAN, -1.0, 2.0]);

    let output = tensor.clamp_max(1.0).into_data().convert::<f32>();
    let values = output.as_slice::<f32>().unwrap();

    assert!(values[0].is_nan());
    assert_eq!(values[1..], [-1.0, 1.0]);
}

// Two-sided clamp still maps NaN to a bound on the cube backends (verified failing on
// CUDA), so this stays on the CPU backends until that is fixed separately.
#[cfg(any(feature = "flex", feature = "ndarray"))]
#[test]
fn clamp_nan_propagation() {
    for dtype in [burn_tensor::DType::F32, burn_tensor::DType::F64] {
        let tensor = TestTensor::<1>::from([f32::NAN, -1.0, 2.0]).cast(dtype);

        let output = tensor.clamp(0.0, 1.0).into_data().convert::<f32>();
        let values = output.as_slice::<f32>().unwrap();

        assert!(values[0].is_nan(), "{dtype:?}");
        assert_eq!(values[1..], [0.0, 1.0], "{dtype:?}");
    }
}

#[cfg(any(feature = "flex", feature = "ndarray"))]
#[test]
fn clamp_min_max_nan_propagation_f64() {
    let tensor = TestTensor::<1>::from([f32::NAN, -1.0, 2.0]).cast(burn_tensor::DType::F64);

    for (output, expected) in [
        (tensor.clone().clamp_min(0.0), [0.0, 2.0]),
        (tensor.clamp_max(1.0), [-1.0, 1.0]),
    ] {
        let output = output.into_data();
        let values = output.as_slice::<f64>().unwrap();

        assert!(values[0].is_nan());
        assert_eq!(values[1..], expected);
    }
}

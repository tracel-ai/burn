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

// A NaN bound makes every element NaN, not just one: x > NaN and x < NaN are both false, so
// each element falls through to the bound. Flex only, ndarray returns the input unchanged.
#[cfg(feature = "flex")]
#[test]
fn clamp_min_nan_bound_propagation() {
    let tensor = TestTensor::<1>::from([-1.0, 0.0, 5.0]);

    let output = tensor.clamp_min(f32::NAN).into_data().convert::<f32>();
    let values = output.as_slice::<f32>().unwrap();

    assert!(values.iter().all(|v| v.is_nan()), "{values:?}");
}

#[cfg(feature = "flex")]
#[test]
fn clamp_max_nan_bound_propagation() {
    let tensor = TestTensor::<1>::from([-1.0, 0.0, 5.0]);

    let output = tensor.clamp_max(f32::NAN).into_data().convert::<f32>();
    let values = output.as_slice::<f32>().unwrap();

    assert!(values.iter().all(|v| v.is_nan()), "{values:?}");
}

// Two-sided clamp used to panic here, since `f32::clamp` rejects a NaN bound. Both
// element types, since flex reaches each one through its own closure.
#[cfg(feature = "flex")]
#[test]
fn clamp_nan_bound_propagation() {
    for dtype in [burn_tensor::DType::F32, burn_tensor::DType::F64] {
        for (min, max) in [(f32::NAN, 1.0), (0.0, f32::NAN), (f32::NAN, f32::NAN)] {
            let tensor = TestTensor::<1>::from([-1.0, 0.0, 5.0]).cast(dtype);

            let output = tensor.clamp(min, max).into_data().convert::<f32>();
            let values = output.as_slice::<f32>().unwrap();

            assert!(
                values.iter().all(|v| v.is_nan()),
                "{dtype:?} {min} {max} -> {values:?}"
            );
        }
    }
}

// Taking the NaN bounds out of the way must leave the native clamp in charge, which
// returns -0.0 rather than the lower bound.
#[cfg(feature = "flex")]
#[test]
fn clamp_keeps_negative_zero() {
    let tensor = TestTensor::<1>::from([-0.0, 0.0]);

    let output = tensor.clamp(0.0, 1.0).into_data().convert::<f32>();
    let values = output.as_slice::<f32>().unwrap();

    assert!(values[0].is_sign_negative(), "{values:?}");
    assert!(values[1].is_sign_positive(), "{values:?}");
}

#[test]
fn clamp_nan_propagation() {
    let tensor = TestTensor::<1>::from([f32::NAN, -1.0, 2.0]);

    let output = tensor.clamp(0.0, 1.0).into_data().convert::<f32>();
    let values = output.as_slice::<f32>().unwrap();

    assert!(values[0].is_nan());
    assert_eq!(values[1..], [0.0, 1.0]);
}

#[cfg(feature = "ndarray")]
#[test]
fn clamp_nan_propagation_through_simd() {
    let mut data = vec![2.0; 64];
    data[0] = f32::NAN;
    let tensor = TestTensor::<1>::from_data(TensorData::new(data, [64]), &Default::default());

    for (output, expected) in [
        (tensor.clone().clamp_min(0.0), [f32::NAN, 2.0]),
        (tensor.clone().clamp_max(1.0), [f32::NAN, 1.0]),
        (tensor.clamp(0.0, 1.0), [f32::NAN, 1.0]),
    ] {
        let values = output.into_data().convert::<f32>();
        let values = values.as_slice::<f32>().unwrap();

        assert!(values[0].is_nan());
        assert!(values[1..].iter().all(|value| *value == expected[1]));
    }
}

#[cfg(feature = "ndarray")]
#[test]
fn clamp_nan_propagation_through_simd_f64() {
    let mut data = vec![2.0_f32; 64];
    data[0] = f32::NAN;
    let tensor = TestTensor::<1>::from_data(TensorData::new(data, [64]), &Default::default())
        .cast(burn_tensor::DType::F64);

    for (output, expected) in [
        (tensor.clone().clamp_min(0.0), 2.0),
        (tensor.clone().clamp_max(1.0), 1.0),
        (tensor.clamp(0.0, 1.0), 1.0),
    ] {
        let values = output.into_data();
        let values = values.as_slice::<f64>().unwrap();

        assert!(values[0].is_nan());
        assert!(values[1..].iter().all(|value| *value == expected));
    }
}

#[cfg(any(feature = "flex", feature = "ndarray"))]
#[test]
fn clamp_min_max_nan_propagation_f64() {
    let tensor = TestTensor::<1>::from([f32::NAN, -1.0, 2.0]).cast(burn_tensor::DType::F64);

    for (output, expected) in [
        (tensor.clone().clamp_min(0.0), [0.0, 2.0]),
        (tensor.clone().clamp_max(1.0), [-1.0, 1.0]),
        (tensor.clamp(0.0, 1.0), [0.0, 1.0]),
    ] {
        let output = output.into_data();
        let values = output.as_slice::<f64>().unwrap();

        assert!(values[0].is_nan());
        assert_eq!(values[1..], expected);
    }
}

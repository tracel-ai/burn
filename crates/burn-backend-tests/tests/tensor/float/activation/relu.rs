use super::*;
use burn_tensor::{TensorData, Tolerance, activation};

#[test]
fn test_relu_d2() {
    let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, -4.0, 5.0]]);

    let output = activation::relu(tensor);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]]), false);
}

#[test]
fn test_relu_d1() {
    let tensor = TestTensor::<1>::from([-2.0, -1.0, 0.0, 1.0, 2.0]);

    let output = activation::relu(tensor);

    output.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.0, 0.0, 0.0, 1.0, 2.0]),
        Tolerance::absolute(1e-6),
    );
}

// All burn backends should propagate NaN through relu. See issue #5609.
#[test]
fn test_relu_nan_propagation() {
    let tensor = TestTensor::<1>::from([f32::NAN, -1.0, 0.0, 2.0]);

    let output = activation::relu(tensor).into_data().convert::<f32>();
    let values = output.as_slice::<f32>().unwrap();

    assert!(values[0].is_nan());
    assert_eq!(values[1..], [0.0, 0.0, 2.0]);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_relu_nan_propagation_through_simd() {
    let mut data = vec![2.0; 64];
    data[0] = f32::NAN;
    let tensor = TestTensor::<1>::from_data(TensorData::new(data, [64]), &Default::default());

    let values = activation::relu(tensor).into_data().convert::<f32>();
    let values = values.as_slice::<f32>().unwrap();

    assert!(values[0].is_nan());
    assert!(values[1..].iter().all(|value| *value == 2.0));
}

#[cfg(feature = "ndarray")]
#[test]
fn test_relu_nan_propagation_through_simd_f64() {
    let mut data = vec![2.0_f32; 64];
    data[0] = f32::NAN;
    let tensor = TestTensor::<1>::from_data(TensorData::new(data, [64]), &Default::default())
        .cast(burn_tensor::DType::F64);

    let values = activation::relu(tensor).into_data();
    let values = values.as_slice::<f64>().unwrap();

    assert!(values[0].is_nan());
    assert!(values[1..].iter().all(|value| *value == 2.0));
}

#[cfg(any(feature = "flex", feature = "ndarray"))]
#[test]
fn test_relu_nan_propagation_f64() {
    let tensor = TestTensor::<1>::from([f32::NAN, -1.0, 0.0, 2.0]).cast(burn_tensor::DType::F64);

    let output = activation::relu(tensor).into_data();
    let values = output.as_slice::<f64>().unwrap();

    assert!(values[0].is_nan());
    assert_eq!(values[1..], [0.0, 0.0, 2.0]);
}

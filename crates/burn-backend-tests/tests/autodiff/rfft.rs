use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::signal;

#[cfg(not(feature = "ndarray"))]
use burn_tensor::{DType, Element};

#[test]
#[cfg(not(feature = "ndarray"))]
fn should_diff_rfft() {
    // Lower precisions not supported
    if !matches!(FloatElem::dtype(), DType::F32 | DType::F64) {
        return;
    }

    let device = AutodiffDevice::new();

    let random1 = TensorData::from([0.26, 0.13, 0.36, 0.24, 0.40, 0.93, 0.12, 0.18]);
    let random2 = TensorData::from([0.03, 0.74, 0.33, 0.70, 0.07, 0.61, 0.32, 0.66]);

    let x = TestTensor::<1>::from_data(random1, &device).require_grad();
    let y = TestTensor::<1>::from_data(random2, &device);

    let (x_re, x_im) = signal::rfft(x.clone(), 0, None);
    let (y_re, y_im) = signal::rfft(y.clone(), 0, None);

    let loss = (x_re * y_re + x_im * y_im).sum();
    let grads = loss.backward();
    let x_grad = x.grad(&grads).unwrap();
    let prod = x_grad.mul(x.inner()).sum();

    TensorData::assert_approx_eq::<FloatElem>(
        &prod.to_data(),
        &loss.to_data(),
        Tolerance::default(),
    );
}

#[test]
#[cfg(not(feature = "ndarray"))]
fn round_trip() {
    if !matches!(FloatElem::dtype(), DType::F32 | DType::F64) {
        return;
    }

    let device = AutodiffDevice::new();

    let random = TensorData::from([
        0.26, 0.13, 0.36, 0.24, 0.40, 0.93, 0.12, 0.18, 0.03, 0.74, 0.33, 0.70, 0.07, 0.61, 0.32,
        0.66,
    ]);

    let tensor = TestTensor::<1>::from_data(random.clone(), &device).require_grad();

    let y = signal::rfft(tensor.clone() * 3.0, 0, None);
    let x = signal::irfft(y.0 * 2.0, y.1 * 2.0, 0, None) / 6.0;

    let loss = x.powi_scalar(2).sum() * 0.5;
    let grads = loss.backward();
    let grad = tensor.grad(&grads).unwrap();

    TensorData::assert_approx_eq::<FloatElem>(&grad.to_data(), &random, Tolerance::default());
}

#[test]
#[cfg(not(feature = "ndarray"))]
fn round_trip_with_dim_nonzero() {
    if !matches!(FloatElem::dtype(), DType::F32 | DType::F64) {
        return;
    }

    let device = AutodiffDevice::new();

    let random = TensorData::from([
        0.26, 0.13, 0.36, 0.24, 0.40, 0.93, 0.12, 0.18, 0.03, 0.74, 0.33, 0.70, 0.07, 0.61, 0.32,
        0.66,
    ]);

    let tensor = TestTensor::<1>::from_data(random.clone(), &device);
    let tensor = tensor.reshape([1, 1, -1, 1, 1]).require_grad();

    let y = signal::rfft(tensor.clone() * 3.0, 2, None);
    let x = signal::irfft(y.0 * 2.0, y.1 * 2.0, 2, None) / 6.0;

    let loss = x.powi_scalar(2).sum() * 0.5;
    let grads = loss.backward();
    let grad = tensor.grad(&grads).unwrap();
    let grad = grad.reshape([-1]);

    TensorData::assert_approx_eq::<FloatElem>(&grad.to_data(), &random, Tolerance::default());
}

#[test]
#[cfg(not(feature = "ndarray"))]
fn round_trip_with_some_n_greater() {
    if !matches!(FloatElem::dtype(), DType::F32 | DType::F64) {
        return;
    }

    let device = AutodiffDevice::new();

    let random = TensorData::from([
        0.26, 0.13, 0.36, 0.24, 0.40, 0.93, 0.12, 0.18, 0.03, 0.74, 0.33, 0.70, 0.07,
    ]);
    let n = Some(16);

    let tensor = TestTensor::<1>::from_data(random.clone(), &device).require_grad();

    let y = signal::rfft(tensor.clone() * 3.0, 0, n);
    let x = signal::irfft(y.0 * 2.0, y.1 * 2.0, 0, n) / 6.0;

    let loss = x.powi_scalar(2).sum() * 0.5;
    let grads = loss.backward();
    let grad = tensor.grad(&grads).unwrap();

    TensorData::assert_approx_eq::<FloatElem>(&grad.to_data(), &random, Tolerance::default());
}

#[test]
#[cfg(not(feature = "ndarray"))]
fn round_trip_with_some_n_less() {
    if !matches!(FloatElem::dtype(), DType::F32 | DType::F64) {
        return;
    }

    let device = AutodiffDevice::new();

    let random = TensorData::from([
        0.26, 0.13, 0.36, 0.24, 0.40, 0.93, 0.12, 0.18, 0.03, 0.74, 0.33, 0.70, 0.07, 0.61, 0.32,
        0.66, 0.16, 0.05, 0.69,
    ]);
    let n = Some(16);

    let tensor = TestTensor::<1>::from_data(random.clone(), &device).require_grad();

    let y = signal::rfft(tensor.clone() * 3.0, 0, n);
    let x = signal::irfft(y.0 * 2.0, y.1 * 2.0, 0, n) / 6.0;

    let loss = x.powi_scalar(2).sum() * 0.5;
    let grads = loss.backward();
    let grad = tensor.grad(&grads).unwrap();

    let lhs = grad.slice_dim(0, 0..16);
    let rhs = tensor.slice_dim(0, 0..16);

    TensorData::assert_approx_eq::<FloatElem>(&lhs.to_data(), &rhs.to_data(), Tolerance::default());
}

#[test]
#[cfg(not(feature = "ndarray"))]
fn round_trip_inverse_with_some_n_greater() {
    if !matches!(FloatElem::dtype(), DType::F32 | DType::F64) {
        return;
    }

    let device = AutodiffDevice::new();

    let random = TensorData::from([0.26, 0.13, 0.36, 0.24, 0.40, 0.93, 0.12]);
    let n = Some(16);

    let tensor = TestTensor::<1>::from_data(random.clone(), &device).require_grad();

    let x = signal::irfft(tensor.clone() * 2.0, tensor.zeros_like(), 0, n) / 6.0;
    let (y, _) = signal::rfft(x * 3.0, 0, n);

    let loss = y.powi_scalar(2).sum() * 0.5;
    let grads = loss.backward();
    let grad = tensor.grad(&grads).unwrap();

    TensorData::assert_approx_eq::<FloatElem>(&grad.to_data(), &random, Tolerance::default());
}

#[test]
#[cfg(not(feature = "ndarray"))]
fn round_trip_inverse_with_some_n_less() {
    if !matches!(FloatElem::dtype(), DType::F32 | DType::F64) {
        return;
    }

    let device = AutodiffDevice::new();

    let random = TensorData::from([
        0.26, 0.13, 0.36, 0.24, 0.40, 0.93, 0.12, 0.18, 0.03, 0.74, 0.33, 0.70,
    ]);
    let n = Some(16);

    let tensor = TestTensor::<1>::from_data(random.clone(), &device).require_grad();

    let x = signal::irfft(tensor.clone() * 2.0, tensor.zeros_like(), 0, n) / 6.0;
    let (y, _) = signal::rfft(x * 3.0, 0, n);

    let loss = y.powi_scalar(2).sum() * 0.5;
    let grads = loss.backward();
    let grad = tensor.grad(&grads).unwrap();

    let lhs = grad.slice_dim(0, 0..9);
    let rhs = tensor.slice_dim(0, 0..9);

    TensorData::assert_approx_eq::<FloatElem>(&lhs.to_data(), &rhs.to_data(), Tolerance::default());
}

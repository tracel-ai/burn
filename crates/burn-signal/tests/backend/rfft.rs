use super::*;
use burn_core::tensor::TensorData;
use burn_core::tensor::Tolerance;
use burn_signal as signal;

#[test]
fn rfft_both_outputs_match_finite_differences() {
    let values = vec![0.2f32, -0.3, 0.7, 0.1, -0.2, 0.6, 0.4, -0.5];
    let device = AutodiffDevice::new();
    let plain_device = burn_core::tensor::Device::default();
    for (dim, n) in [(0, Some(2)), (0, Some(8)), (1, None)] {
        let objective = |input| {
            let (real, imag) = signal::rfft(input, dim, n);
            real.square().sum() + imag.square().sum() * 0.7
        };
        let input = TestTensor::<2>::from_data(TensorData::new(values.clone(), [4, 2]), &device)
            .require_grad();
        let gradients = objective(input.clone()).backward();
        let actual = input.grad(&gradients).unwrap().into_data();
        let expected: Vec<f32> = (0..values.len())
            .map(|i| {
                let mut plus = values.clone();
                let mut minus = values.clone();
                plus[i] += 1e-3;
                minus[i] -= 1e-3;
                let eval = |data| {
                    objective(TestTensor::<2>::from_data(
                        TensorData::new(data, [4, 2]),
                        &plain_device,
                    ))
                    .into_scalar::<f32>()
                };
                (eval(plus) - eval(minus)) / 2e-3
            })
            .collect();
        actual.assert_approx_eq::<f32>(
            &TensorData::new(expected, [4, 2]),
            Tolerance::absolute(3e-3),
        );
    }
}

#[test]
fn irfft_both_inputs_match_finite_differences() {
    let real_values = vec![0.2f32, -0.3, 0.7, 0.1, -0.2, 0.6];
    let imag_values = vec![0.4f32, 0.5, -0.6, 0.3, 0.8, -0.7];
    let device = AutodiffDevice::new();
    let plain_device = burn_core::tensor::Device::default();
    for n in [Some(2), None, Some(8)] {
        let objective = |real, imag| signal::irfft(real, imag, 0, n).square().sum();
        let real =
            TestTensor::<2>::from_data(TensorData::new(real_values.clone(), [3, 2]), &device)
                .require_grad();
        let imag =
            TestTensor::<2>::from_data(TensorData::new(imag_values.clone(), [3, 2]), &device)
                .require_grad();
        let gradients = objective(real.clone(), imag.clone()).backward();
        for (component, actual) in [
            real.grad(&gradients).unwrap(),
            imag.grad(&gradients).unwrap(),
        ]
        .into_iter()
        .enumerate()
        {
            let expected: Vec<f32> = (0..6)
                .map(|i| {
                    let eval = |delta| {
                        let mut re = real_values.clone();
                        let mut im = imag_values.clone();
                        if component == 0 {
                            re[i] += delta;
                        } else {
                            im[i] += delta;
                        }
                        objective(
                            TestTensor::<2>::from_data(TensorData::new(re, [3, 2]), &plain_device),
                            TestTensor::<2>::from_data(TensorData::new(im, [3, 2]), &plain_device),
                        )
                        .into_scalar::<f32>()
                    };
                    (eval(1e-3) - eval(-1e-3)) / 2e-3
                })
                .collect();
            actual.into_data().assert_approx_eq::<f32>(
                &TensorData::new(expected, [3, 2]),
                Tolerance::absolute(1e-3),
            );
        }
    }
}

#[test]
fn stft_istft_round_trip_preserves_gradients() {
    let input = TestTensor::<2>::from_data(
        [[0.2, -0.3, 0.7, 0.1, -0.2, 0.6, 0.4, -0.5]],
        &AutodiffDevice::new(),
    )
    .require_grad();
    for center in [false, true] {
        let options = signal::StftOptions {
            n_fft: 4,
            hop_length: 2,
            center,
            ..signal::StftOptions::new(4)
        };
        let spectrum = signal::stft(input.clone(), None, options);
        let output = signal::istft(spectrum, None, Some(8), options);
        let gradients = output.sum().backward();
        input
            .grad(&gradients)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[1.0f32; 8]]), Tolerance::absolute(1e-5));
    }
}

#[cfg(not(feature = "ndarray"))]
use burn_core::tensor::{DType, Element};

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

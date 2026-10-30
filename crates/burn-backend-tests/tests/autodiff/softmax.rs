use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

fn assert_high_confidence_softmax_grad(dtype: burn_tensor::DType) {
    // Independent f64 references for d softmax(z)[0] / dz, with z = [0, -gap, -gap].
    let (gap, tail_grad, relative_error) = match dtype {
        burn_tensor::DType::F16 => (9.0, 1.2334890671462635e-4, 1e-2),
        burn_tensor::DType::BF16 => (8.0, 3.350129398148664e-4, 2e-2),
        burn_tensor::DType::F32 => (18.0, 1.5229978816903543e-8, 1e-5),
        burn_tensor::DType::F64 => (40.0, 4.248354255291589e-18, 1e-12),
        _ => unreachable!(),
    };
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data([[gap / 2.0, -gap / 2.0, -gap / 2.0]], &device)
        .cast(dtype)
        .require_grad();
    let selected = activation::softmax(tensor.clone(), 1).slice([0..1, 0..1]);
    let grads = selected.sum().backward();
    let grad = tensor.grad(&grads).unwrap().into_data();
    let values: Vec<f64> = grad.iter::<f64>().collect();

    assert!(
        values[0] > 0.0,
        "{dtype:?}: dominant probability gradient vanished: {values:?}"
    );
    grad.convert::<f64>().assert_approx_eq::<f64>(
        &TensorData::from([[2.0 * tail_grad, -tail_grad, -tail_grad]]),
        Tolerance::relative(relative_error),
    );
    let total: f64 = values.iter().sum();
    let magnitude: f64 = values.iter().map(|value| value.abs()).sum();
    assert!(
        total.abs() <= relative_error * magnitude,
        "gradient sum {total}: {values:?}"
    );
}

#[test]
fn test_softmax_high_confidence_grad() {
    assert_high_confidence_softmax_grad(<FloatElem as burn_tensor::Element>::dtype());
}

#[cfg(feature = "flex")]
#[test]
fn test_softmax_high_confidence_grad_bf16() {
    assert_high_confidence_softmax_grad(burn_tensor::DType::BF16);
}

#[cfg(feature = "flex")]
#[test]
fn test_softmax_high_confidence_grad_f16() {
    assert_high_confidence_softmax_grad(burn_tensor::DType::F16);
}

#[cfg(feature = "flex")]
#[test]
fn test_softmax_high_confidence_grad_f64() {
    assert_high_confidence_softmax_grad(burn_tensor::DType::F64);
}

#[test]
fn test_log_softmax_rotated_high_confidence_grad() {
    let (score, tail, relative_error) =
        if <FloatElem as burn_tensor::Element>::dtype() == burn_tensor::DType::F16 {
            (4.5, 1.2337935164345024e-4, 1e-2)
        } else {
            (9.0, 1.522997928080808e-8, 1e-5)
        };
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data(
        [
            [score, -score, -score],
            [-score, score, -score],
            [-score, -score, score],
        ],
        &device,
    )
    .require_grad();
    let indices = TestTensorInt::<2>::from_data([[0], [1], [2]], &device);
    let loss = activation::log_softmax(tensor.clone(), 1)
        .gather(1, indices)
        .sum()
        .neg();
    let grads = loss.backward();
    tensor
        .grad(&grads)
        .unwrap()
        .into_data()
        .convert::<f64>()
        .assert_approx_eq::<f64>(
            &TensorData::from([
                [-2.0 * tail, tail, tail],
                [tail, -2.0 * tail, tail],
                [tail, tail, -2.0 * tail],
            ]),
            Tolerance::relative(relative_error),
        );
}

#[test]
fn test_log_softmax_tied_maxima_grad() {
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<2>::from_data([[2.0, 2.0, 0.0], [-2.0, 1.0, 1.0]], &device).require_grad();
    let indices = TestTensorInt::<2>::from_data([[0], [2]], &device);
    let loss = activation::log_softmax(tensor.clone(), 1)
        .gather(1, indices)
        .sum()
        .neg();
    let grads = loss.backward();
    tensor
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [-0.5316894691665188, 0.4683105308334812, 0.06337893833303762],
                [
                    0.024288897679263205,
                    0.4878555511603684,
                    -0.5121444488396316,
                ],
            ]),
            Tolerance::default(),
        );
}

#[test]
fn test_log_softmax_transposed_non_last_axis_grad() {
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<2>::from_data([[1.0, 2.0, 4.0], [-1.0, 3.0, 0.0]], &device).require_grad();
    let indices = TestTensorInt::<2>::from_data([[0, 0]], &device);
    let loss = activation::log_softmax(tensor.clone().transpose(), 0)
        .gather(0, indices)
        .sum()
        .neg();
    let grads = loss.backward();
    tensor
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([
                [-0.9579899338659339, 0.11419519938459449, 0.8437947344813395],
                [-0.9828521744544796, 0.9362395518765058, 0.0466126225779739],
            ]),
            Tolerance::default(),
        );
}

fn assert_masked_activation_grad(logarithmic: bool, dtype: burn_tensor::DType) {
    let half = dtype == burn_tensor::DType::F16;
    let (score, log_tail, soft_tail, correction, relative_error) = if half {
        (
            4.5,
            1.2339457598623172e-4,
            1.233793497648489e-4,
            1.2340218972325883e-4,
            1e-2,
        )
    } else {
        (
            9.0,
            1.522997951276035e-8,
            1.5229979280808076e-8,
            1.5229979628736488e-8,
            1e-5,
        )
    };
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data(
        [
            [score, -score, f64::NEG_INFINITY],
            [0.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
        ],
        &device,
    )
    .cast(dtype)
    .require_grad();
    let output = if logarithmic {
        activation::log_softmax(tensor.clone(), 1)
    } else {
        activation::softmax(tensor.clone(), 1)
    };
    let selected = output.slice([0..2, 0..1]);
    if logarithmic {
        selected
            .clone()
            .into_data()
            .convert::<f64>()
            .assert_approx_eq::<f64>(
                &TensorData::from([[-correction], [0.0]]),
                Tolerance::relative(relative_error),
            );
    } else {
        selected
            .clone()
            .into_data()
            .convert::<f64>()
            .assert_approx_eq::<f64>(
                &TensorData::from([[1.0 - log_tail], [1.0]]),
                Tolerance::relative(relative_error),
            );
    }
    // Only finite outputs enter the scalar objective; masked log probabilities are -inf.
    let grads = selected.sum().neg().backward();
    let grad = tensor.grad(&grads).unwrap().into_data().convert::<f64>();
    let tail = if logarithmic { log_tail } else { soft_tail };
    grad.assert_approx_eq::<f64>(
        &TensorData::from([[-tail, tail, 0.0], [0.0, 0.0, 0.0]]),
        Tolerance::relative(relative_error),
    );
}

#[test]
fn test_log_softmax_masked_classes_grad() {
    assert_masked_activation_grad(true, <FloatElem as burn_tensor::Element>::dtype());
}

#[test]
fn test_softmax_masked_classes_grad() {
    assert_masked_activation_grad(false, <FloatElem as burn_tensor::Element>::dtype());
}

#[cfg(feature = "flex")]
#[test]
fn test_log_softmax_masked_classes_grad_f16() {
    assert_masked_activation_grad(true, burn_tensor::DType::F16);
}

#[cfg(feature = "flex")]
#[test]
fn test_softmax_masked_classes_grad_f16() {
    assert_masked_activation_grad(false, burn_tensor::DType::F16);
}

#[test]
fn test_softmax_grad() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::<2>::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = activation::softmax(tensor_3, 1).matmul(tensor_2.clone());

    let grads = tensor_4.backward();
    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[1.179665, 1.179661], [0.005462, 0.005463]]);

    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(0.05, 0.5));

    let expected = TensorData::from([[0.253469, 0.286237], [0.528630, 2.931664]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(0.05, 0.05));
}

#[test]
fn test_log_softmax_grad() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::<2>::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = activation::log_softmax(tensor_3, 1).matmul(tensor_2.clone());

    let grads = tensor_4.backward();
    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[-4.3939, -4.3939], [-12.9709, -12.9709]]);
    // f16 gradients from log-softmax + matmul amplify error, so we increase the tolerance
    // to account for limited precision and large representable step sizes in this range.
    let tolerance = Tolerance::permissive().set_half_precision_relative(6e-2);

    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[30.5984, -47.2267], [55.9631, -56.5914]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn test_quiet_softmax_grad() {
    let data = TensorData::from([
        [f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::NEG_INFINITY, 0.0],
    ]);
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data(data, &device).require_grad();

    let output = activation::quiet_softmax(tensor.clone(), 1);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    assert!(!grad.clone().contains_nan().into_scalar::<bool>());
    let expected = TensorData::from([[0.0, 0.0], [0.0, 0.25]]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

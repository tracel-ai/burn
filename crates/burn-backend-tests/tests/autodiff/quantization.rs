use super::*;
use burn_tensor::{TensorData, Tolerance};

use burn_tensor::quantization::QuantValue;

/// Dequantizing on an autodiff device yields a leaf that requires no gradient:
/// the packed base is frozen, so composing it with tracked tensors propagates
/// gradients through the tracked side only. This is the shape a low-rank
/// adapter over a quantized base (QLoRA) relies on.
#[test]
fn should_diff_composition_over_dequantized_base() {
    let device = AutodiffDevice::new();
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    // Last dim is a multiple of 4 so packed quantized stores can hold it.
    let w = TestTensor::<2>::from_data(
        [
            [0.5, -1.0, 0.25, 2.0],
            [1.5, 0.0, -0.75, 1.0],
            [-2.0, 0.5, 1.25, -0.5],
            [0.75, -1.5, 0.0, 1.75],
        ],
        &device,
    );
    let x_data = TensorData::from([[1.0, -2.0, 0.5, 1.5], [0.0, 1.0, -1.0, 2.0]]);
    let a_data = TensorData::from([[0.5, -0.25], [1.0, 0.75], [-0.5, 0.25], [0.25, 1.0]]);
    let b_data = TensorData::from([[1.0, 0.5, -0.5, 0.25], [-0.25, 1.5, 0.75, -1.0]]);

    // The quantized composition: x @ (dequantize(quantize(w)) + a @ b).
    let w_q = w.quantize_dynamic(&scheme);
    let w_dq = w_q.dequantize();

    let x = TestTensor::<2>::from_data(x_data.clone(), &device).require_grad();
    let a = TestTensor::<2>::from_data(a_data.clone(), &device).require_grad();
    let b = TestTensor::<2>::from_data(b_data.clone(), &device).require_grad();

    let y = x
        .clone()
        .matmul(w_dq.clone().add(a.clone().matmul(b.clone())));
    let grads = y.sum().backward();

    let grad_x = x
        .grad(&grads)
        .expect("x is tracked through the composition");
    let grad_a = a.grad(&grads).expect("the a factor is tracked");
    let grad_b = b.grad(&grads).expect("the b factor is tracked");
    assert!(
        w_dq.grad(&grads).is_none(),
        "the dequantized base is frozen: no gradient may reach it"
    );

    // Reference: the identical computation with the dequantized values as a
    // plain (non-quantized) leaf must produce the same gradients.
    let w_ref = TestTensor::<2>::from_data(w_dq.to_data(), &device);
    let x_ref = TestTensor::<2>::from_data(x_data, &device).require_grad();
    let a_ref = TestTensor::<2>::from_data(a_data, &device).require_grad();
    let b_ref = TestTensor::<2>::from_data(b_data, &device).require_grad();

    let y_ref = x_ref
        .clone()
        .matmul(w_ref.add(a_ref.clone().matmul(b_ref.clone())));
    let grads_ref = y_ref.sum().backward();

    let tolerance = Tolerance::default();
    grad_x
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
    grad_a
        .to_data()
        .assert_approx_eq::<FloatElem>(&a_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
    grad_b
        .to_data()
        .assert_approx_eq::<FloatElem>(&b_ref.grad(&grads_ref).unwrap().to_data(), tolerance);
}

/// A tracked float times a quantized weight, without dequantizing by hand:
/// the mixed `q_matmul` runs on the autodiff backend and the gradient flows
/// through the float operand.
#[test]
fn should_diff_matmul_with_quantized_rhs() {
    let device = AutodiffDevice::new();
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let w = TestTensor::<2>::from_data(
        [
            [0.5, -1.0, 0.25, 2.0],
            [1.5, 0.0, -0.75, 1.0],
            [-2.0, 0.5, 1.25, -0.5],
            [0.75, -1.5, 0.0, 1.75],
        ],
        &device,
    );
    let x_data = TensorData::from([[1.0, -2.0, 0.5, 1.5], [0.0, 1.0, -1.0, 2.0]]);

    let w_q = w.quantize_dynamic(&scheme);
    let x = TestTensor::<2>::from_data(x_data.clone(), &device).require_grad();

    let y = x.clone().matmul(w_q.clone());
    let grads = y.sum().backward();
    let grad_x = x.grad(&grads).expect("x is tracked through q_matmul");

    // Reference: the same product against the dequantized weight.
    let w_ref = TestTensor::<2>::from_data(w_q.dequantize().to_data(), &device);
    let x_ref = TestTensor::<2>::from_data(x_data, &device).require_grad();
    let grads_ref = x_ref.clone().matmul(w_ref).sum().backward();

    grad_x.to_data().assert_approx_eq::<FloatElem>(
        &x_ref.grad(&grads_ref).unwrap().to_data(),
        Tolerance::default(),
    );
}

/// The round trip itself on an autodiff device: quantize a tracked tensor,
/// dequantize it back, and read the values. Quantization detaches, so this
/// neither panics nor grows the graph.
#[test]
fn should_quantize_dequantize_on_autodiff_device() {
    let device = AutodiffDevice::new();
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let data = TensorData::from([[-1.8, -1.0, 0.0, 0.5], [0.25, 1.0, 1.5, -0.75]]);
    let tensor = TestTensor::<2>::from_data(data.clone(), &device).require_grad();

    let dequantized = tensor.quantize_dynamic(&scheme).dequantize();

    dequantized
        .to_data()
        .assert_approx_eq::<FloatElem>(&data, Tolerance::rel_abs(1e-1, 1e-1));
}

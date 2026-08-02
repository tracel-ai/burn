use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
#[should_panic]
fn panic_on_second_backward_without_retain() {
    let device = AutodiffDevice::new();

    let a = TestTensor::<1>::from_data(TensorData::from([3.0_f32, 4.0]), &device).require_grad();

    let b = TestTensor::<1>::from_data(TensorData::from([5.0_f32, 6.0]), &device).require_grad();

    let result = a.clone().reshape([2, 1]).matmul(b.clone().reshape([1, 2]));

    let result1 = result.clone().slice([0..1, 0..1]).sum();
    let result2 = result.clone().slice([1..2, 0..1]).sum();

    let _grads1 = result1.backward();
    let _grads2 = result2.backward();
}

#[test]
fn produce_same_gradients_on_repeated_backward() {
    // Two retain passes on the same graph must yield identical gradients.
    let device = AutodiffDevice::new();

    let x =
        TestTensor::<1>::from_data(TensorData::from([1.0_f32, 2.0, 3.0]), &device).require_grad();

    let y = x.clone().powf_scalar(2.0).sum();

    let grads1 = y.backward_retain();
    let grads2 = y.backward_retain();

    let grad1 = x.grad(&grads1).unwrap();
    let grad2 = x.grad(&grads2).unwrap();

    TensorData::assert_approx_eq::<FloatElem>(
        &grad1.to_data(),
        &grad2.to_data(),
        Tolerance::default(),
    );
}

#[test]
fn match_standard_backward_gradients() {
    // A retain graph backward must produce the same gradients as a standard backward.
    let device = AutodiffDevice::new();

    let x_retain =
        TestTensor::<1>::from_data(TensorData::from([1.0_f32, 2.0, 3.0]), &device).require_grad();

    let x_standard =
        TestTensor::<1>::from_data(TensorData::from([1.0_f32, 2.0, 3.0]), &device).require_grad();

    let y_retain = x_retain.clone().mul_scalar(3.0_f32).sum();
    let y_standard = x_standard.clone().mul_scalar(3.0_f32).sum();

    let grads_retain = y_retain.backward_retain();
    let grads_standard = y_standard.backward();

    let grad_retain = x_retain.grad(&grads_retain).unwrap();
    let grad_standard = x_standard.grad(&grads_standard).unwrap();

    TensorData::assert_approx_eq::<FloatElem>(
        &grad_retain.to_data(),
        &grad_standard.to_data(),
        Tolerance::default(),
    );
}

#[test]
fn allow_multiple_backward_after_retain() {
    // Graph survives retain passes and still runs a final consuming backward.
    let device = AutodiffDevice::new();

    let x = TestTensor::<1>::from_data(TensorData::from([2.0_f32, 3.0]), &device).require_grad();

    let y = x.clone().mul_scalar(2.0_f32).sum();
    let expected = TensorData::from([2.0_f32, 2.0]);

    let grads1 = y.backward_retain();
    let grad1 = x.grad(&grads1).unwrap();
    TensorData::assert_approx_eq::<FloatElem>(&grad1.to_data(), &expected, Tolerance::default());

    let grads2 = y.backward_retain();
    let grad2 = x.grad(&grads2).unwrap();
    TensorData::assert_approx_eq::<FloatElem>(&grad2.to_data(), &expected, Tolerance::default());

    let grads3 = y.backward();
    let grad3 = x.grad(&grads3).unwrap();
    TensorData::assert_approx_eq::<FloatElem>(&grad3.to_data(), &expected, Tolerance::default());
}

#[test]
fn retain_graph_across_multiple_outputs() {
    // Multiple backward passes over slices of a shared output.
    let device = AutodiffDevice::new();

    let a = TestTensor::<1>::from_data(TensorData::from([3.0_f32, 4.0]), &device).require_grad();

    let b = TestTensor::<1>::from_data(TensorData::from([5.0_f32, 6.0]), &device);

    let result = a.clone().reshape([2, 1]).matmul(b.reshape([1, 2]));

    let result1 = result.clone().slice([0..1, 0..1]).sum();
    let result2 = result.clone().slice([1..2, 0..1]).sum();

    let grads1 = result1.backward_retain();
    let grads2 = result2.backward_retain();

    let grad1 = a.grad(&grads1).unwrap();
    let grad2 = a.grad(&grads2).unwrap();

    TensorData::assert_approx_eq::<FloatElem>(
        &grad1.to_data(),
        &TensorData::from([5.0_f32, 0.0]),
        Tolerance::default(),
    );
    TensorData::assert_approx_eq::<FloatElem>(
        &grad2.to_data(),
        &TensorData::from([0.0_f32, 5.0]),
        Tolerance::default(),
    );
}

#[test]
fn retain_graph_with_deeper_computation() {
    // Retain works correctly through a deeper graph, two retain passes must agree
    let device = AutodiffDevice::new();

    let x =
        TestTensor::<1>::from_data(TensorData::from([1.0_f32, 2.0, 3.0]), &device).require_grad();

    let y = x.clone().powf_scalar(2.0).sin().sum();

    let grads1 = y.backward_retain();
    let grads2 = y.backward_retain();

    let grad1 = x.grad(&grads1).unwrap();
    let grad2 = x.grad(&grads2).unwrap();

    TensorData::assert_approx_eq::<FloatElem>(
        &grad1.to_data(),
        &grad2.to_data(),
        Tolerance::default(),
    );
}

#[test]
fn retain_graph_with_two_inputs() {
    // Retain correctly computes gradients for both inputs independently.
    let device = AutodiffDevice::new();

    let x =
        TestTensor::<1>::from_data(TensorData::from([1.0_f32, 2.0, 3.0]), &device).require_grad();

    let z =
        TestTensor::<1>::from_data(TensorData::from([4.0_f32, 5.0, 6.0]), &device).require_grad();

    let y = x.clone().mul(z.clone()).sum();

    let grads1 = y.backward_retain();
    let grads2 = y.backward_retain();

    let grad_x1 = x.grad(&grads1).unwrap();
    let grad_x2 = x.grad(&grads2).unwrap();
    let grad_z1 = z.grad(&grads1).unwrap();
    let grad_z2 = z.grad(&grads2).unwrap();

    TensorData::assert_approx_eq::<FloatElem>(
        &grad_x1.to_data(),
        &TensorData::from([4.0_f32, 5.0, 6.0]),
        Tolerance::default(),
    );
    TensorData::assert_approx_eq::<FloatElem>(
        &grad_x2.to_data(),
        &grad_x1.to_data(),
        Tolerance::default(),
    );
    TensorData::assert_approx_eq::<FloatElem>(
        &grad_z1.to_data(),
        &TensorData::from([1.0_f32, 2.0, 3.0]),
        Tolerance::default(),
    );
    TensorData::assert_approx_eq::<FloatElem>(
        &grad_z2.to_data(),
        &grad_z1.to_data(),
        Tolerance::default(),
    );
}

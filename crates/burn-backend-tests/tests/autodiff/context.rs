//! Runtime tests for merging concrete and autodiff dispatch contexts.

use super::*;
use burn_tensor::{GradientCheckpointingStrategy, TensorData};

#[test]
fn gradient_flows_to_enabled_operand_but_not_disabled_constant() {
    let x = TestTensor::<1>::from_floats([2.0, 3.0], &AutodiffDevice::new());
    assert!(x.is_autodiff());
    assert!(!x.is_tracked());

    let x = x.require_grad();
    let constant =
        TestTensor::<1>::from_floats([4.0, 5.0], &AutodiffDevice::new().without_autodiff());

    let output = (x.clone() * constant.clone()).sum();
    assert!(output.is_autodiff());
    assert!(output.is_tracked());

    let grads = output.backward();
    x.grad(&grads)
        .expect("the enabled operand should receive a gradient")
        .into_data()
        .assert_eq(&TensorData::from([4.0f32, 5.0]), false);

    // Context merging is operation-local: the concrete operand is neither mutated nor added to
    // the graph, and remains unavailable as a gradient target.
    assert!(!constant.is_autodiff());
    assert!(!constant.is_tracked());
    assert!(!constant.is_require_grad());
    assert!(constant.grad(&grads).is_none());
}

#[test]
fn autodiff_and_tracking_states_follow_graph_transitions() {
    let plain = TestTensor::<1>::from_floats([1.0, 2.0], &AutodiffDevice::new().without_autodiff());
    assert!(!plain.is_autodiff());
    assert!(!plain.is_tracked());
    assert!(!plain.is_require_grad());

    let autodiff = plain.autodiff();
    assert!(autodiff.is_autodiff());
    assert!(!autodiff.is_tracked());
    assert!(!autodiff.is_require_grad());

    let leaf = autodiff.require_grad();
    assert!(leaf.is_autodiff());
    assert!(leaf.is_tracked());
    assert!(leaf.is_require_grad());

    // Detaching a leaf preserves its gradient-retention setting and starts a new tracked leaf.
    let detached_leaf = leaf.clone().detach();
    assert!(detached_leaf.is_autodiff());
    assert!(detached_leaf.is_tracked());
    assert!(detached_leaf.is_require_grad());

    let derived = leaf.mul_scalar(2.0);
    assert!(derived.is_autodiff());
    assert!(derived.is_tracked());
    assert!(!derived.is_require_grad());

    // A detached non-leaf stays in the autodiff context but has no recorded graph.
    let detached = derived.clone().detach();
    assert!(detached.is_autodiff());
    assert!(!detached.is_tracked());
    assert!(!detached.is_require_grad());

    let plain = derived.without_autodiff();
    assert!(!plain.is_autodiff());
    assert!(!plain.is_tracked());
    assert!(!plain.is_require_grad());
}

#[test]
fn tracked_state_does_not_report_consumed_tape_availability() {
    let leaf = TestTensor::<1>::from_floats([1.0, 2.0], &AutodiffDevice::new()).require_grad();
    let output = leaf.mul_scalar(2.0).sum();

    assert!(output.is_tracked());
    let _ = output.backward();

    // Tracking is a property of the node and remains observable after backward consumes its tape.
    assert!(output.is_tracked());
}

#[test]
fn tensor_autodiff_conversions_are_idempotent() {
    let plain = TestTensor::<1>::from_floats([1.0, 2.0], &AutodiffDevice::new().without_autodiff());

    let plain = plain.without_autodiff().inner();
    assert!(!plain.is_autodiff());

    let autodiff = plain.autodiff();
    assert!(autodiff.is_autodiff());

    let autodiff = TestTensor::from_inner(autodiff);
    assert!(autodiff.is_autodiff());

    let plain = autodiff.without_autodiff().inner();
    assert!(!plain.is_autodiff());
}

#[test]
fn enabling_autodiff_twice_preserves_checkpointing_strategy() {
    let device = AutodiffDevice::new();
    let strategy = device.gradient_checkpointing_strategy();
    let tensor = TestTensor::<1>::from_floats([1.0, 2.0], &device);

    let tensor = tensor.autodiff();
    assert_eq!(tensor.gradient_checkpointing_strategy(), strategy);

    let tensor = TestTensor::from_inner(tensor);
    assert_eq!(tensor.gradient_checkpointing_strategy(), strategy);
}

#[test]
fn device_autodiff_conversions_are_idempotent() {
    let device = AutodiffDevice::new();
    let strategy = device.gradient_checkpointing_strategy();
    let device = device.autodiff();

    assert!(device.is_autodiff());
    assert_eq!(device.gradient_checkpointing_strategy(), strategy);

    let device = device.without_autodiff().inner();
    assert!(!device.is_autodiff());
    assert_eq!(device.gradient_checkpointing_strategy(), None);
}

#[test]
fn tensor_autodiff_builder_configures_checkpointing() {
    let tensor =
        TestTensor::<1>::from_floats([1.0, 2.0], &AutodiffDevice::new().without_autodiff())
            .autodiff()
            .with_gradient_checkpointing_strategy(GradientCheckpointingStrategy::Balanced);

    assert!(tensor.is_autodiff());
    assert_eq!(
        tensor.gradient_checkpointing_strategy(),
        Some(GradientCheckpointingStrategy::Balanced)
    );
}

#[test]
#[should_panic(expected = "Tensor::with_gradient_checkpointing_strategy requires autodiff")]
fn tensor_checkpointing_strategy_setter_requires_autodiff() {
    let _ = TestTensor::<1>::from_floats([1.0, 2.0], &AutodiffDevice::new().without_autodiff())
        .with_gradient_checkpointing_strategy(GradientCheckpointingStrategy::Balanced);
}

#[test]
#[should_panic(expected = "Tensor::require_grad requires autodiff")]
fn requiring_gradients_on_a_plain_tensor_is_rejected() {
    let _ = TestTensor::<1>::ones([2], &AutodiffDevice::new().without_autodiff()).require_grad();
}

#[test]
#[should_panic(expected = "Tensor::require_grad requires autodiff")]
fn enabling_gradient_retention_on_a_plain_tensor_is_rejected() {
    let _ = TestTensor::<1>::ones([2], &AutodiffDevice::new().without_autodiff())
        .set_require_grad(true);
}

#[test]
fn disabling_gradients_on_a_plain_tensor_is_harmless() {
    let tensor = TestTensor::<1>::ones([2], &AutodiffDevice::new().without_autodiff())
        .set_require_grad(false);
    assert!(!tensor.is_autodiff());
    assert!(!tensor.is_require_grad());
}

#[test]
fn untracked_alias_cannot_read_or_remove_a_leaf_gradient() {
    let constant = TestTensor::<1>::ones([2], &AutodiffDevice::new());
    let leaf = constant.clone().require_grad();
    let mut grads = leaf.clone().mul_scalar(2.0).sum().backward();
    assert!(!constant.is_tracked());
    assert!(constant.grad(&grads).is_none());
    assert!(constant.grad_remove(&mut grads).is_none());
    leaf.grad_remove(&mut grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([2.0f32, 2.0]), false);
}

#[test]
#[should_panic(expected = "Tensor::backward requires a tracked autodiff tensor")]
fn backward_rejects_an_untracked_operation() {
    let constant = TestTensor::<1>::ones([2], &AutodiffDevice::new()).mul_scalar(2.0);
    assert!(!constant.is_tracked());
    let _ = constant.backward();
}

#[cfg(feature = "quantization")]
#[test]
fn quantized_tensors_never_retain_gradients() {
    use burn_tensor::quantization::QuantValue;
    let device = AutodiffDevice::new();
    for device in [device.clone().without_autodiff(), device] {
        let scheme = device
            .settings()
            .quantization
            .scheme
            .with_value(QuantValue::Q8S);
        let tensor =
            TestTensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &device).quantize_dynamic(&scheme);
        let dtype = tensor.dtype();
        let tensor = tensor
            .require_grad()
            .set_require_grad(true)
            .set_require_grad(false);
        assert_eq!(tensor.dtype(), dtype);
        assert_eq!(tensor.is_autodiff(), device.is_autodiff());
        assert!(!tensor.is_tracked());
        assert!(!tensor.is_require_grad());
    }
}

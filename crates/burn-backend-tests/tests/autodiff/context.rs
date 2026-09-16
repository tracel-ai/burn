//! Runtime tests for merging concrete and autodiff dispatch contexts.
//! Included in both checkpointing variants of the shared autodiff suite, using
//! the backend selected by `Device::default()` / `BURN_DEVICE`.

use super::*;
use burn_tensor::{Device, GradientCheckpointingStrategy, TensorData};

#[test]
fn gradient_flows_to_enabled_operand_but_not_disabled_constant() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<1>::from_floats([2.0, 3.0], &device);
    assert!(x.is_autodiff());
    assert!(!x.is_tracked());

    let x = x.require_grad();
    let constant = TestTensor::<1>::from_floats([4.0, 5.0], &Device::default());

    let output = (x.clone() * constant.clone()).sum();
    assert!(output.is_autodiff());
    assert!(output.is_tracked());
    assert_eq!(
        output.gradient_checkpointing_strategy(),
        device.gradient_checkpointing_strategy()
    );

    let grads = output.backward();
    x.grad(&grads)
        .expect("the enabled operand should receive a gradient")
        .into_data()
        .assert_eq(
            &TensorData::from([4.0f32, 5.0]).convert::<FloatElem>(),
            true,
        );

    // Context merging is operation-local: the concrete operand is neither mutated nor added to
    // the graph, and remains unavailable as a gradient target.
    assert!(!constant.is_autodiff());
    assert!(!constant.is_tracked());
    assert!(!constant.is_require_grad());
    assert!(constant.grad(&grads).is_none());
}

#[test]
fn autodiff_and_tracking_states_follow_graph_transitions() {
    let strategy = AutodiffDevice::new()
        .gradient_checkpointing_strategy()
        .unwrap();
    let plain = TestTensor::<1>::from_floats([1.0, 2.0], &Device::default());
    assert!(!plain.is_autodiff());
    assert!(!plain.is_tracked());
    assert!(!plain.is_require_grad());

    let autodiff = plain
        .autodiff()
        .with_gradient_checkpointing_strategy(strategy);
    assert!(autodiff.is_autodiff());
    assert!(!autodiff.is_tracked());
    assert!(!autodiff.is_require_grad());
    assert_eq!(autodiff.gradient_checkpointing_strategy(), Some(strategy));

    let leaf = autodiff.require_grad();
    assert!(leaf.is_autodiff());
    assert!(leaf.is_tracked());
    assert!(leaf.is_require_grad());

    // Detaching a leaf preserves its gradient-retention setting and starts a new tracked leaf.
    let detached_leaf = leaf.clone().detach();
    assert!(detached_leaf.is_autodiff());
    assert!(detached_leaf.is_tracked());
    assert!(detached_leaf.is_require_grad());
    assert_eq!(
        detached_leaf.gradient_checkpointing_strategy(),
        Some(strategy)
    );

    let derived = leaf.mul_scalar(2.0);
    assert!(derived.is_autodiff());
    assert!(derived.is_tracked());
    assert!(!derived.is_require_grad());
    assert_eq!(derived.gradient_checkpointing_strategy(), Some(strategy));

    // A detached non-leaf stays in the autodiff context but has no recorded graph.
    let detached = derived.clone().detach();
    assert!(detached.is_autodiff());
    assert!(!detached.is_tracked());
    assert!(!detached.is_require_grad());
    assert_eq!(detached.gradient_checkpointing_strategy(), Some(strategy));

    let plain = derived.without_autodiff();
    assert!(!plain.is_autodiff());
    assert!(!plain.is_tracked());
    assert!(!plain.is_require_grad());
    assert_eq!(plain.gradient_checkpointing_strategy(), None);
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
    let plain = TestTensor::<1>::from_floats([1.0, 2.0], &Device::default());

    let plain = plain.without_autodiff().inner();
    assert!(!plain.is_autodiff());

    let autodiff = plain.autodiff();
    assert!(autodiff.is_autodiff());
    // Enabling a plain tensor uses the default strategy, regardless of the
    // checkpointing variant used by the surrounding test suite.
    assert_eq!(
        autodiff.gradient_checkpointing_strategy(),
        Some(GradientCheckpointingStrategy::Disabled)
    );

    let autodiff = TestTensor::from_inner(autodiff);
    assert!(autodiff.is_autodiff());
    assert_eq!(
        autodiff.gradient_checkpointing_strategy(),
        Some(GradientCheckpointingStrategy::Disabled)
    );

    let plain = autodiff.without_autodiff().inner();
    assert!(!plain.is_autodiff());
}

#[test]
fn enabling_autodiff_twice_preserves_checkpointing_strategy() {
    let device = AutodiffDevice::new();
    let expected = device.gradient_checkpointing_strategy();
    let tensor = TestTensor::<1>::from_floats([1.0, 2.0], &device);

    let tensor = tensor.autodiff();
    assert_eq!(tensor.gradient_checkpointing_strategy(), expected);

    let tensor = TestTensor::from_inner(tensor);
    assert_eq!(tensor.gradient_checkpointing_strategy(), expected);
}

#[test]
fn device_autodiff_conversions_are_idempotent() {
    let device = AutodiffDevice::new();
    let expected = device.gradient_checkpointing_strategy();
    let device = device.autodiff();

    assert!(device.is_autodiff());
    assert_eq!(device.gradient_checkpointing_strategy(), expected);

    let device = device.without_autodiff().inner();
    assert!(!device.is_autodiff());
    assert_eq!(device.gradient_checkpointing_strategy(), None);
}

#[test]
fn tensor_autodiff_builder_configures_checkpointing() {
    let strategy = AutodiffDevice::new()
        .gradient_checkpointing_strategy()
        .unwrap();
    let tensor = TestTensor::<1>::from_floats([1.0, 2.0], &Device::default())
        .autodiff()
        .with_gradient_checkpointing_strategy(strategy);

    assert!(tensor.is_autodiff());
    assert_eq!(tensor.gradient_checkpointing_strategy(), Some(strategy));
}

#[test]
#[should_panic(expected = "Tensor::with_gradient_checkpointing_strategy requires autodiff")]
fn tensor_checkpointing_strategy_setter_requires_autodiff() {
    let _ = TestTensor::<1>::from_floats([1.0, 2.0], &Device::default())
        .with_gradient_checkpointing_strategy(GradientCheckpointingStrategy::Balanced);
}

#[test]
#[should_panic(expected = "Tensor::require_grad requires autodiff")]
fn requiring_gradients_on_a_plain_tensor_is_rejected() {
    let _ = TestTensor::<1>::ones([2], &Device::default()).require_grad();
}

#[test]
#[should_panic(expected = "Tensor::require_grad requires autodiff")]
fn enabling_gradient_retention_on_a_plain_tensor_is_rejected() {
    let _ = TestTensor::<1>::ones([2], &Device::default()).set_require_grad(true);
}

#[test]
fn disabling_gradients_on_a_plain_tensor_is_harmless() {
    let tensor = TestTensor::<1>::ones([2], &Device::default()).set_require_grad(false);
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
    leaf.grad_remove(&mut grads).unwrap().into_data().assert_eq(
        &TensorData::from([2.0f32, 2.0]).convert::<FloatElem>(),
        true,
    );
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
#[cfg(feature = "quantization")]
fn quantized_tensors_never_retain_gradients() {
    use burn_tensor::quantization::QuantValue;
    for device in [Device::default(), AutodiffDevice::new()] {
        let scheme = device
            .settings()
            .quantization
            .scheme
            .with_value(QuantValue::Q8S);
        // Packed quantized stores require a last dimension divisible by four.
        let tensor =
            TestTensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], &device).quantize_dynamic(&scheme);
        let dtype = tensor.dtype();
        let tensor = tensor
            .require_grad()
            .set_require_grad(true)
            .set_require_grad(false);
        assert_eq!(tensor.dtype(), dtype);
        assert_eq!(tensor.is_autodiff(), device.is_autodiff());
        assert_eq!(
            tensor.gradient_checkpointing_strategy(),
            device.gradient_checkpointing_strategy()
        );
        assert!(!tensor.is_tracked());
        assert!(!tensor.is_require_grad());
    }
}

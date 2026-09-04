//! Runtime tests for merging concrete and autodiff dispatch contexts.

#![cfg(all(feature = "autodiff", feature = "flex"))]

use burn::tensor::{Device, GradientCheckpointingStrategy, Tensor, TensorData};

#[test]
fn gradient_flows_to_enabled_operand_but_not_disabled_constant() {
    let x = Tensor::<1>::from_floats([2.0, 3.0], &Device::flex().autodiff());
    assert!(x.is_autodiff());
    assert!(!x.is_tracked());

    let x = x.require_grad();
    let constant = Tensor::<1>::from_floats([4.0, 5.0], &Device::flex());

    let output = (x.clone() * constant.clone()).sum();
    assert!(output.is_autodiff());
    assert!(output.is_tracked());

    let grads = output.backward();
    x.grad(&grads)
        .expect("the enabled operand should receive a gradient")
        .into_data()
        .assert_eq(&TensorData::from([4.0f32, 5.0]), true);

    // Context merging is operation-local: the concrete operand is neither mutated nor added to
    // the graph, and remains unavailable as a gradient target.
    assert!(!constant.is_autodiff());
    assert!(!constant.is_tracked());
    assert!(!constant.is_require_grad());
    assert!(constant.grad(&grads).is_none());
}

#[test]
fn autodiff_and_tracking_states_follow_graph_transitions() {
    let plain = Tensor::<1>::from_floats([1.0, 2.0], &Device::flex());
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
fn tensor_autodiff_conversions_are_idempotent() {
    let plain = Tensor::<1>::from_floats([1.0, 2.0], &Device::flex());

    let plain = plain.without_autodiff().inner();
    assert!(!plain.is_autodiff());

    let autodiff = plain.autodiff();
    assert!(autodiff.is_autodiff());

    let autodiff = Tensor::from_inner(autodiff);
    assert!(autodiff.is_autodiff());

    let plain = autodiff.without_autodiff().inner();
    assert!(!plain.is_autodiff());
}

#[test]
fn enabling_autodiff_twice_preserves_checkpointing_strategy() {
    let device = Device::flex().autodiff().gradient_checkpointing();
    let tensor = Tensor::<1>::from_floats([1.0, 2.0], &device);

    let tensor = tensor.autodiff();
    assert_eq!(
        tensor.gradient_checkpointing_strategy(),
        Some(GradientCheckpointingStrategy::Balanced)
    );

    let tensor = Tensor::from_inner(tensor);
    assert_eq!(
        tensor.gradient_checkpointing_strategy(),
        Some(GradientCheckpointingStrategy::Balanced)
    );
}

#[test]
fn device_autodiff_conversions_are_idempotent() {
    let device = Device::flex()
        .autodiff()
        .gradient_checkpointing()
        .autodiff();

    assert!(device.is_autodiff());
    assert_eq!(
        device.gradient_checkpointing_strategy(),
        Some(GradientCheckpointingStrategy::Balanced)
    );

    let device = device.without_autodiff().inner();
    assert!(!device.is_autodiff());
    assert_eq!(device.gradient_checkpointing_strategy(), None);
}

#[test]
fn tensor_autodiff_builder_configures_checkpointing() {
    let tensor = Tensor::<1>::from_floats([1.0, 2.0], &Device::flex())
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
    let _ = Tensor::<1>::from_floats([1.0, 2.0], &Device::flex())
        .with_gradient_checkpointing_strategy(GradientCheckpointingStrategy::Balanced);
}

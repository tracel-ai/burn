//! Runtime tests for merging concrete and autodiff dispatch contexts.

#![cfg(all(feature = "autodiff", feature = "flex"))]

use burn::tensor::{Device, GradientCheckpointingStrategy, Tensor, TensorData};

#[test]
fn gradient_flows_to_enabled_operand_but_not_disabled_constant() {
    let x = Tensor::<1>::from_floats([2.0, 3.0], &Device::flex().autodiff()).require_grad();
    let constant = Tensor::<1>::from_floats([4.0, 5.0], &Device::flex());

    let output = (x.clone() * constant.clone()).sum();
    assert!(output.device().is_autodiff());

    let grads = output.backward();
    x.grad(&grads)
        .expect("the enabled operand should receive a gradient")
        .into_data()
        .assert_eq(&TensorData::from([4.0f32, 5.0]), true);

    // Context merging is operation-local: the concrete operand is neither mutated nor added to
    // the graph, and remains unavailable as a gradient target.
    assert!(!constant.device().is_autodiff());
    assert!(!constant.is_require_grad());
}

#[test]
fn tensor_autodiff_conversions_are_idempotent() {
    let plain = Tensor::<1>::from_floats([1.0, 2.0], &Device::flex());

    let plain = plain.without_autodiff().inner();
    assert!(!plain.device().is_autodiff());

    let autodiff = plain.autodiff();
    assert!(autodiff.device().is_autodiff());

    let autodiff = Tensor::from_inner(autodiff);
    assert!(autodiff.device().is_autodiff());

    let plain = autodiff.without_autodiff().inner();
    assert!(!plain.device().is_autodiff());
}

#[test]
fn enabling_autodiff_twice_preserves_checkpointing_strategy() {
    let device = Device::flex().autodiff().gradient_checkpointing();
    let tensor = Tensor::<1>::from_floats([1.0, 2.0], &device);

    let tensor = tensor.autodiff();
    assert_eq!(
        tensor.device().gradient_checkpointing_strategy(),
        GradientCheckpointingStrategy::Balanced
    );

    let tensor = Tensor::from_inner(tensor);
    assert_eq!(
        tensor.device().gradient_checkpointing_strategy(),
        GradientCheckpointingStrategy::Balanced
    );
}

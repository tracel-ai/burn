//! Runtime tests for merging concrete and autodiff dispatch contexts.

#![cfg(all(feature = "autodiff", feature = "flex"))]

use burn::tensor::{Device, Tensor, TensorData};

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

//! Backward validates consumed intermediates without rejecting reusable parameter leaves.

use super::*;
use burn_tensor::TensorData;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn assert_consumed(f: impl FnOnce()) {
    let error = catch_unwind(AssertUnwindSafe(f)).expect_err("consumed ancestry must be rejected");
    let message = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(
        message.contains("graph tape has already been consumed"),
        "unexpected panic: {message}"
    );
}

#[test]
fn backward_rejects_a_new_loss_with_consumed_ancestors() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
    let derived = x.clone().mul_scalar(2.0);

    let grads = derived.clone().sum().backward();
    x.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([2.0f32, 2.0]), false);

    // A fresh loss node still depends on the consumed intermediate. Tracking describes
    // graph participation, not tape availability.
    assert!(derived.is_tracked());
    let second_loss = derived.mul_scalar(3.0).sum();
    assert_consumed(|| {
        let _ = second_loss.backward();
    });
}

#[test]
fn consumed_graph_is_rejected_before_an_incorrect_parameter_update() {
    let device = AutodiffDevice::new();
    let mut parameter = TestTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
    let shared = parameter.clone() * parameter.clone();
    let _ = shared.clone().sum().backward();

    // The forward loss is still correct, despite the consumed x^2 backward step.
    let loss = (shared.mul_scalar(3.0) + parameter.clone()).sum();
    loss.to_data()
        .assert_eq(&TensorData::from([44.0f32]), false);

    // For loss = sum(3x^2 + x), dloss/dx = 6x + 1 = [13, 19]. Silently
    // skipping x^2 would return [1, 1] and an incorrect SGD update [1.75, 2.75].
    assert_consumed(|| {
        let grads = loss.backward();
        let grad = parameter.grad(&grads).unwrap();
        parameter = parameter.clone().without_autodiff() - grad.mul_scalar(0.25);
    });
    parameter
        .to_data()
        .assert_eq(&TensorData::from([2.0f32, 3.0]), false);

    // Recomputing the forward restores the complete derivative and permits the update.
    let shared = parameter.clone() * parameter.clone();
    let grads = (shared.mul_scalar(3.0) + parameter.clone())
        .sum()
        .backward();
    let grad = parameter.grad(&grads).unwrap();
    grad.to_data()
        .assert_eq(&TensorData::from([13.0f32, 19.0]), false);
    parameter = parameter.without_autodiff() - grad.mul_scalar(0.25);
    parameter
        .into_data()
        .assert_eq(&TensorData::from([-1.25f32, -1.75]), false);
}

#[test]
fn rejection_preserves_a_fresh_branch() {
    let device = AutodiffDevice::new();
    for stale_first in [false, true] {
        let x = TestTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
        let stale = x.clone().mul_scalar(2.0);
        let _ = stale.clone().sum().backward();
        let fresh = x.clone() * x.clone();
        let invalid = if stale_first {
            stale + fresh.clone()
        } else {
            fresh.clone() + stale
        }
        .sum();
        assert_consumed(|| {
            let _ = invalid.backward();
        });
        let grads = fresh.sum().backward();
        x.grad(&grads)
            .unwrap()
            .into_data()
            .assert_eq(&TensorData::from([4.0f32, 6.0]), false);
    }
}

#[test]
fn existing_losses_cannot_reuse_a_shared_intermediate() {
    let device = AutodiffDevice::new();
    for keep_shared_handle in [false, true] {
        let x = TestTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
        let shared = x.clone() * x.clone();
        let first = shared.clone().sum();
        let second = shared.clone().mul_scalar(3.0).sum();
        let _shared_handle = keep_shared_handle.then_some(shared);

        let _ = first.backward();
        assert_consumed(|| {
            let _ = second.backward();
        });
    }
}

#[test]
fn shared_losses_can_be_combined_before_backward() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
    let shared = x.clone() * x.clone();
    let first = shared.clone().sum();
    let second = shared.mul_scalar(3.0).sum();
    let grads = (first + second).backward();
    x.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([16.0f32, 24.0]), false);
}

#[test]
fn detaching_consumed_ancestry_starts_a_new_lineage() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
    let shared = x.clone() * x.clone();
    let _ = shared.clone().sum().backward();

    let new_leaf = shared.detach().require_grad();
    let grads = new_leaf.clone().mul_scalar(3.0).sum().backward();
    assert!(x.grad(&grads).is_none());
    new_leaf
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([3.0f32, 3.0]), false);
}

#[test]
fn fresh_forwards_can_reuse_a_leaf() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<1>::ones([2], &device).require_grad();
    for scale in [2.0f32, 3.0] {
        let grads = x.clone().mul_scalar(scale).sum().backward();
        x.grad(&grads)
            .unwrap()
            .into_data()
            .assert_eq(&TensorData::from([scale, scale]), false);
    }
}

#[test]
fn reused_leaves_can_be_parents_alongside_deeper_nodes() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<1>::ones([2], &device).require_grad();
    let _ = x.clone().sum().backward();
    // Keep the branch alive throughout construction to isolate ancestry validation from
    // concurrent orphan sweeps in other tests.
    let first = x.clone().mul_scalar(2.0);
    let deeper = first.clone().mul_scalar(3.0);
    let grads = (deeper.clone() + x.clone()).sum().backward();
    x.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([7.0f32, 7.0]), false);

    // Keep the branch alive throughout construction to isolate ancestry validation from
    // concurrent orphan sweeps in other tests.
    let first = x.clone().mul_scalar(2.0);
    let deeper = first.clone().mul_scalar(3.0);
    let grads = TestTensor::cat(vec![x.clone(), deeper.clone(), x.clone()], 0)
        .sum()
        .backward();
    x.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([8.0f32, 8.0]), false);
}

#[test]
fn existing_branches_can_share_only_a_leaf_even_after_its_handle_is_dropped() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<1>::ones([2], &device).require_grad();
    let y = TestTensor::<1>::ones([2], &device).require_grad();
    let first = x.clone().mul_scalar(2.0);
    let second = x * y.clone();
    let _ = first.sum().backward();
    let grads = second.sum().backward();
    y.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([1.0f32, 1.0]), false);
}

#[test]
fn shared_intermediates_and_cat_accumulate_once_per_edge() {
    let device = AutodiffDevice::new();
    let x = TestTensor::<1>::ones([2], &device).require_grad();
    let shared = x.clone().mul_scalar(2.0);
    let combined = TestTensor::cat(vec![shared.clone(), shared.clone().mul_scalar(3.0)], 0);
    let grads = combined.sum().backward();
    x.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([8.0f32, 8.0]), false);
    assert_consumed(|| {
        let _ = TestTensor::cat(vec![shared.clone(), shared], 0)
            .sum()
            .backward();
    });
}

#[test]
fn backward_twice_on_the_same_output_is_rejected() {
    let device = AutodiffDevice::new();
    let output = TestTensor::<1>::ones([2], &device).require_grad().sum();
    let clone = output.clone();
    let _ = output.backward();
    assert_consumed(|| {
        let _ = output.backward();
    });
    assert_consumed(|| {
        let _ = clone.backward();
    });
}

// NdArray conservatively reports can_mut() as false even for unique storage.
#[cfg(not(feature = "ndarray"))]
#[test]
fn orphan_cleanup_releases_saved_buffers_with_a_reusable_leaf_still_alive() {
    let device = AutodiffDevice::new();
    let leaf = TestTensor::<1>::ones([2], &device).require_grad();
    let _ = leaf.clone().sum().backward();
    let saved = TestTensor::<1>::from_floats([2.0, 3.0], &device.clone().without_autodiff());
    assert!(saved.can_mut());
    let abandoned = leaf.clone() * saved.clone();
    assert!(
        !saved.can_mut(),
        "the unused backward step must hold a buffer reference"
    );

    drop(abandoned);

    // Sweep orphaned work through a completely unrelated backward graph.
    let _ = TestTensor::<1>::ones([2], &device)
        .require_grad()
        .sum()
        .backward();
    assert!(
        saved.can_mut(),
        "orphan cleanup must release the saved buffer despite the live leaf"
    );
    let grads = leaf.clone().sum().backward();
    assert!(leaf.grad(&grads).is_some());
}

#[cfg(not(feature = "ndarray"))]
#[test]
fn cleanup_preserves_live_saved_buffers_and_releases_them_after_backward() {
    let device = AutodiffDevice::new();
    let leaf = TestTensor::<1>::ones([2], &device).require_grad();
    let saved = TestTensor::<1>::from_floats([2.0, 3.0], &device.clone().without_autodiff());
    let output = leaf.clone() * saved.clone();
    assert!(!saved.can_mut());

    let _ = TestTensor::<1>::ones([2], &device)
        .require_grad()
        .sum()
        .backward();
    assert!(!saved.can_mut(), "a live branch must keep its saved buffer");
    let grads = output.sum().backward();
    leaf.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([2.0f32, 3.0]), false);
    drop(grads);
    assert!(
        saved.can_mut(),
        "consumed steps must release their saved buffers"
    );
}

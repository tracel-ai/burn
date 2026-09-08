//! Gradients and tensor context across compute-backend transfers.
#![cfg(all(feature = "autodiff", feature = "flex", feature = "ndarray"))]
#![allow(deprecated)]

use burn::tensor::{Device, GradientCheckpointingStrategy, Tensor, TensorData};

fn devices() -> [(Device, Device); 2] {
    [
        (Device::flex(), Device::ndarray()),
        (Device::ndarray(), Device::flex()),
    ]
}

fn strategies() -> [GradientCheckpointingStrategy; 2] {
    [
        GradientCheckpointingStrategy::Disabled,
        GradientCheckpointingStrategy::Balanced,
    ]
}

fn enabled(tensor: Tensor<1>, strategy: GradientCheckpointingStrategy) -> Tensor<1> {
    tensor
        .autodiff()
        .with_gradient_checkpointing_strategy(strategy)
}

#[test]
fn transfers_preserve_values_and_source_context() {
    for (source, destination) in devices() {
        for strategy in strategies() {
            let plain = Tensor::<1>::from_floats([2.0, 3.0], &source);
            let opposite = match strategy {
                GradientCheckpointingStrategy::Disabled => {
                    destination.clone().autodiff().gradient_checkpointing()
                }
                GradientCheckpointingStrategy::Balanced => destination.clone().autodiff(),
            };
            for target in [&destination, &opposite] {
                let moved_plain = plain.clone().to_device(target);
                assert!(!moved_plain.is_autodiff());
                assert!(!moved_plain.is_tracked());
                assert_eq!(moved_plain.device(), destination);
                moved_plain
                    .into_data()
                    .assert_eq(&TensorData::from([2.0f32, 3.0]), true);

                for tracked in [false, true] {
                    let input = enabled(plain.clone(), strategy).set_require_grad(tracked);
                    let dtype = input.dtype();
                    let output = input.to_device(target);
                    assert_eq!(output.device(), destination);
                    assert_eq!(output.shape().dims(), [2]);
                    assert_eq!(output.dtype(), dtype);
                    assert!(output.is_autodiff());
                    assert_eq!(output.gradient_checkpointing_strategy(), Some(strategy));
                    assert_eq!(output.is_tracked(), tracked);
                    assert!(!output.is_require_grad());
                    output
                        .into_data()
                        .assert_eq(&TensorData::from([2.0f32, 3.0]), true);
                }
            }
        }
    }
}

#[test]
fn backward_crosses_transfer_and_accumulates_local_and_remote_branches() {
    for (source, destination) in devices() {
        for strategy in strategies() {
            let x = enabled(Tensor::<1>::from_floats([2.0, 3.0], &source), strategy).require_grad();
            let before = x.clone() * x.clone();
            let moved = before.to_device(&destination);
            // A local x^2 branch and a transferred x^4 branch join on the destination.
            let local = (x.clone() * x.clone()).to_device(&destination);
            let output = (moved.clone() * moved.clone() + local).sum();
            let grads = output.backward();
            let grad = x
                .grad(&grads)
                .expect("source gradient must survive the transfer");
            assert_eq!(grad.device(), source);
            assert!(!grad.is_autodiff());
            grad.into_data()
                .assert_eq(&TensorData::from([36.0f32, 114.0]), true);
            assert!(moved.grad(&grads).is_none());
        }
    }
}

#[test]
fn round_trip_and_untracked_transferred_constants_support_checkpointing() {
    for (source, destination) in devices() {
        for strategy in strategies() {
            let x = enabled(Tensor::<1>::from_floats([2.0, 3.0], &source), strategy).require_grad();
            // This untracked transfer must remain available as a saved value in backward.
            let constant = enabled(Tensor::<1>::from_floats([4.0, 5.0], &source), strategy)
                .to_device(&destination);
            let moved = x.clone().to_device(&destination);
            let squared = moved.clone() * moved;
            let remote = squared * constant.clone();
            let returned = remote.to_device(&source);
            let output = (returned + x.clone()).sum();
            let grads = output.backward();
            x.grad(&grads)
                .unwrap()
                .into_data()
                .assert_eq(&TensorData::from([17.0f32, 31.0]), true);
            assert!(!constant.is_tracked());
            assert!(constant.grad(&grads).is_none());
        }
    }
}

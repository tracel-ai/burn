#![no_std] // Must keep it for testing

use burn::module::RunningState;
use burn::nn::BatchNormConfig;
use burn::tensor::{Device, Tensor, TensorData, Tolerance};

/// Training a batch norm updates its running statistics, which records the update per thread.
/// Without `std` there is no thread id, so this used to panic.
#[test]
fn batch_norm_training_updates_the_running_mean() {
    let device = Device::flex().autodiff();
    let module = BatchNormConfig::new(3).init(&device);
    let input = Tensor::<3>::from_floats(
        [
            [[0.9601, 0.7277], [0.6272, 0.9034], [0.9378, 0.7230]],
            [[0.6356, 0.1362], [0.0249, 0.9509], [0.6600, 0.5945]],
        ],
        &device,
    );

    let _output = module.forward(input.clone());

    // With a zero initial mean and the default momentum of 0.1, the running mean is a tenth
    // of the batch mean of each channel.
    module
        .running_mean
        .value_sync()
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::from([0.061490, 0.062660, 0.072883]),
            Tolerance::default(),
        );

    // A second update is folded into the first one: 0.9 * 0.1 + 0.1 = 0.19 times the batch mean.
    let _output = module.forward(input);
    module
        .running_mean
        .value_sync()
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::from([0.116831, 0.119054, 0.138478]),
            Tolerance::default(),
        );
}

/// An update recorded while the previous one is still pending folds that one into the value first,
/// as both come from the same (only) thread.
#[test]
fn running_state_folds_a_pending_update_into_the_value() {
    let device = Device::flex();
    let state = RunningState::new(Tensor::<1>::from_floats([0.0, 0.0], &device));

    state.update(Tensor::from_floats([1.0, 2.0], &device));
    state.update(Tensor::from_floats([3.0, 4.0], &device));

    // The second update folded the first one, and is still pending itself
    state
        .value()
        .into_data()
        .assert_eq(&TensorData::from([1.0f32, 2.0]), false);
    state
        .value_sync()
        .into_data()
        .assert_eq(&TensorData::from([3.0f32, 4.0]), false);
}

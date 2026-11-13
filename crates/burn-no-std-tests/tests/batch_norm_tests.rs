#![no_std] // Must keep it for testing

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

    let _output = module.forward(input);

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
}

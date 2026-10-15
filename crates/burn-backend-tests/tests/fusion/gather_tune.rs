//! Tests that tuning a fused gather never reads through indices a discarded launch left.

use super::*;
use burn_tensor::TensorData;
use cubek::cubecl::execution::{ProcessMode, ProcessModeOverride, StatisticsCollector};

/// Under `CompileAndAutotune` the launch that would write the indices is discarded, so they hold
/// whatever their memory held. Uploaded far out of range, they stand in for that garbage: the
/// tune of the gather fused into a normalizing reduce must not read through them.
#[test]
fn tuning_a_fused_gather_never_reads_through_unwritten_indices() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let [rows, experts, routes] = [128, 32, 4];
        let scores = TestTensor::<2>::from_data(
            TensorData::new(vec![1.0f32; rows * experts], [rows, experts]),
            &device,
        );
        let indices = TestTensorInt::<2>::from_data(
            TensorData::new(vec![i32::MAX; rows * routes], [rows, routes]),
            &device,
        );
        device.sync().unwrap();

        let collector = StatisticsCollector::new();
        let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
        let weights = scores.gather(1, indices);
        let _normalized = weights.clone() / weights.sum_dim(1).add_scalar(1e-6);
        device.sync().expect("the tune reads only in-range indices");
        drop(tune);
    });
}

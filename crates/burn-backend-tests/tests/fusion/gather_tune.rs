//! Tests that tuning a fused gather never reads through indices a discarded launch left.

use super::*;
use cubek::cubecl::execution::{ProcessMode, ProcessModeOverride, StatisticsCollector};

#[test]
fn tuning_a_fused_gather_never_reads_through_unwritten_indices() {
    test_stream().executes(|| {
        let device = Default::default();
        let scores = TestTensor::<2>::from_floats([[1.0; 32]; 128], &device);
        let indices = TestTensorInt::<2>::from_ints([[i32::MAX; 4]; 128], &device);
        device.sync().unwrap();

        let collector = StatisticsCollector::new();
        let _tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
        let weights = scores.gather(1, indices);
        let _normalized = weights.clone() / weights.sum_dim(1).add_scalar(1e-6);
        device.sync().expect("the tune reads only in-range indices");
    });
}

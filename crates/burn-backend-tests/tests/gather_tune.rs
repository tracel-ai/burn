//! Tests that tuning a fused gather never reads through indices a discarded launch left.
//!
//! Kept in its own test binary because `CompileAndAutotune` discards ordinary
//! launches on every thread and device, including those of unrelated tests.

#![cfg(all(feature = "cube", feature = "fusion"))]

pub type FloatElem = f32;
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;

use backend::*;
use burn_tensor::ProcessMode;
use burn_tensor::StreamId;

#[test]
fn tuning_a_fused_gather_never_reads_through_unwritten_indices() {
    StreamId::allocate().executes(|| {
        let device = Default::default();
        let scores = TestTensor::<2>::from_floats([[1.0; 32]; 128], &device);
        let indices = TestTensorInt::<2>::from_ints([[i32::MAX; 4]; 128], &device);
        device.sync().unwrap();

        let tune = device
            .process_mode(ProcessMode::CompileAndAutotune)
            .unwrap();
        let weights = scores.gather(1, indices);
        let _normalized = weights.clone() / weights.sum_dim(1).add_scalar(1e-6);
        device.sync().expect("the tune reads only in-range indices");
        tune.close().unwrap();
    });
}

//! Graph capture/replay integration tests for the closure-based
//! [`capture`](burn_tensor::capture) API.
//!
//! On a backend with hardware graph support (CUDA/HIP) the capture records the
//! closure's launches and every replay is a single dispatch against the
//! original buffers; elsewhere replay falls back to re-running the closure.
//! Both paths must produce the eager result.
//!
//! Isolated in this test binary: `capture` arms device-global allocation state
//! (the persistent pool) between `graph_prepare` and `stop_capture`, so it must
//! not interleave with unrelated tests allocating on the same device. Tests are
//! additionally `#[serial]` so two captures never overlap.

#![cfg(feature = "cube")]

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

use burn_tensor::{Device, Tolerance};
use serial_test::serial;

/// Replaying a captured pure closure reproduces the eager result, repeatedly.
///
/// Safety of the `replay` calls: the closure owns a clone of `input` (keeping
/// every captured buffer alive as long as the graph), and all replays and
/// output reads happen sequentially on this thread — nothing else touches the
/// graph's tensors.
#[test]
#[serial]
fn capture_replay_matches_eager() {
    let device = Device::default();
    let input = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let expected = input.clone().mul_scalar(2.0).add_scalar(1.0).into_data();

    let mut graph = burn_tensor::capture(&device, || input.clone().mul_scalar(2.0).add_scalar(1.0));

    for _ in 0..3 {
        let out = unsafe { graph.replay() }.clone().into_data();
        out.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
    }
}

/// The output handle is stable across replays on the hardware path: `output()`
/// returns the same tensor whose buffer each replay overwrites.
#[test]
#[serial]
fn capture_output_is_stable() {
    let device = Device::default();
    let input = TestTensor::<1>::from_data([1.0, 2.0, 3.0, 4.0], &device);
    let expected = input.clone().add_scalar(10.0).into_data();

    let mut graph = burn_tensor::capture(&device, || input.clone().add_scalar(10.0));

    // Safety: see `capture_replay_matches_eager`.
    unsafe { graph.replay() };
    graph
        .output()
        .clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

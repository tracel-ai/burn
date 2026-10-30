//! Tests of the execution modes a device opens: [`Device::process_mode`] and
//! [`Device::stream_mode`].
//!
//! Kept in its own test binary because a process mode that discards launches
//! applies to the whole process on the cubecl runtimes, including the launches
//! of unrelated tests; the tests are `#[serial]` for the same reason. Run with
//! the `remote` feature, they open the modes on the server.

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;

use backend::*;
use burn_tensor::{Device, ProcessMode, StreamMode, Tolerance};
use serial_test::serial;

/// A step put back to executing inside a pass that discards launches runs, and
/// the pass counts the kernels it reached.
#[test]
#[serial]
fn a_stream_in_execute_runs_inside_a_discarding_pass() {
    let device = Device::default();
    let input = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    device.sync().unwrap();

    let gather = device.process_mode(ProcessMode::CompileOnly).unwrap();
    assert!(
        gather.applies() || !cfg!(feature = "cube"),
        "a cubecl device applies its modes"
    );
    if !gather.applies() {
        return;
    }
    let reader = gather.reader();
    let execute = device.stream_mode(StreamMode::Execute).unwrap();
    let output = input.clone().mul_scalar(3.5).add_scalar(-2.0).into_data();
    execute.close().unwrap();
    let statistics = gather.close().unwrap();

    output.assert_approx_eq::<FloatElem>(
        &TestTensor::<2>::from_data([[1.5, 5.0], [8.5, 12.0]], &device).into_data(),
        Tolerance::default(),
    );
    assert!(
        statistics.compilation.registered > 0,
        "the pass reached the step's kernels: {statistics:?}"
    );
    assert!(
        reader.statistics().is_err(),
        "a closed mode has no statistics to read"
    );
}

/// One mode is open at a time where it applies; opening the mode already open
/// joins it, and it stays open until the last guard closes.
#[test]
#[serial]
fn another_mode_is_refused_while_one_is_open() {
    let device = Device::default();

    let tune = device
        .process_mode(ProcessMode::CompileAndAutotune)
        .unwrap();
    if !tune.applies() {
        return;
    }
    assert!(device.process_mode(ProcessMode::CompileOnly).is_err());

    let joined = device
        .process_mode(ProcessMode::CompileAndAutotune)
        .unwrap();
    tune.close().unwrap();
    joined.reader().statistics().unwrap();
    joined.close().unwrap();

    // Closed, so another mode opens.
    device
        .process_mode(ProcessMode::CompileOnly)
        .unwrap()
        .close()
        .unwrap();
}

/// A dropped guard closes its mode: launches run again after it.
#[test]
#[serial]
fn a_dropped_guard_closes_its_mode() {
    let device = Device::default();
    let input = TestTensor::<1>::from_data([1.0, 2.0], &device);

    let gather = device.process_mode(ProcessMode::CompileOnly).unwrap();
    core::mem::drop(gather);

    input
        .add_scalar(1.0)
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TestTensor::<1>::from_data([2.0, 3.0], &device).into_data(),
            Tolerance::default(),
        );
}

//! Tests of the execution modes a device opens with [`Device::mode`].
//!
//! Kept in its own test binary because a process mode that discards launches
//! applies to the whole process on the cubecl runtimes, including the launches
//! of unrelated tests; the tests are `#[serial]` for the same reason. Run with
//! the `remote` feature, they open the modes on the server.
//!
//! A kernel counts as registered only the first time the process meets it, so
//! each test counts a computation no other test runs.

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;

use backend::*;
use burn_tensor::{Device, ProcessMode, StreamMode, Tolerance};
use serial_test::serial;

/// Whether `device` applies execution modes, asserting that a cubecl device
/// does: a test on a backend that compiles nothing has nothing to check.
fn applies_modes(device: &Device) -> bool {
    let probe = device.mode(ProcessMode::CompileOnly).unwrap();
    let applies = probe.applies();
    probe.close().unwrap();
    assert!(
        applies || !cfg!(feature = "cube"),
        "a cubecl device applies its modes"
    );
    applies
}

/// A step put back to executing inside a pass that discards launches runs, and
/// the pass counts the kernels it reached.
#[test]
#[serial]
fn a_stream_in_execute_runs_inside_a_discarding_pass() {
    let device = Device::default();
    if !applies_modes(&device) {
        return;
    }
    let input = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    device.sync().unwrap();

    let compile = device.mode(ProcessMode::CompileOnly).unwrap();
    let reader = compile.reader();
    let execute = device.mode(StreamMode::Execute).unwrap();
    let output = input.clone().mul_scalar(3.5).add_scalar(-2.0).into_data();
    execute.close().unwrap();
    let statistics = compile.close().unwrap();

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

/// Work registered before the mode opens runs, even when a backend still holds
/// it queued: it was registered to execute.
#[test]
#[serial]
fn work_registered_before_the_mode_opens_runs() {
    let device = Device::default();
    if !applies_modes(&device) {
        return;
    }
    let input = TestTensor::<1>::from_data([0.0, 1.0], &device);
    device.sync().unwrap();
    let before = input.exp();

    let compile = device.mode(ProcessMode::CompileOnly).unwrap();
    compile.close().unwrap();

    before.into_data().assert_approx_eq::<FloatElem>(
        &TestTensor::<1>::from_data([1.0, core::f32::consts::E], &device).into_data(),
        Tolerance::default(),
    );
}

/// Work registered under the mode is counted by it, even when a backend still
/// holds it queued at the close, and even when another thread closes it.
#[test]
#[serial]
fn work_registered_under_the_mode_counts_whoever_closes_it() {
    let device = Device::default();
    if !applies_modes(&device) {
        return;
    }
    let input = TestTensor::<1>::from_data([0.5, 1.5], &device);
    device.sync().unwrap();

    let compile = device.mode(ProcessMode::CompileOnly).unwrap();
    let _discarded = input.tanh().sqrt();
    let statistics = std::thread::spawn(move || compile.close().unwrap())
        .join()
        .unwrap();

    assert!(
        statistics.compilation.registered > 0,
        "the queued work ran under the mode: {statistics:?}"
    );
}

/// One process mode is open at a time where it applies; opening the mode
/// already open joins it, and it stays open until the last guard closes.
#[test]
#[serial]
fn another_mode_is_refused_while_one_is_open() {
    let device = Device::default();
    if !applies_modes(&device) {
        return;
    }

    let tune = device.mode(ProcessMode::CompileAndAutotune).unwrap();
    assert!(device.mode(ProcessMode::CompileOnly).is_err());

    let joined = device.mode(ProcessMode::CompileAndAutotune).unwrap();
    tune.close().unwrap();
    joined.reader().statistics().unwrap();
    assert!(
        device.mode(ProcessMode::CompileOnly).is_err(),
        "the joined guard still holds the mode open"
    );
    joined.close().unwrap();

    device
        .mode(ProcessMode::CompileOnly)
        .unwrap()
        .close()
        .unwrap();
}

/// A dropped guard closes its mode: launches run again after it.
#[test]
#[serial]
fn a_dropped_guard_closes_its_mode() {
    let device = Device::default();
    if !applies_modes(&device) {
        return;
    }
    let input = TestTensor::<1>::from_data([1.0, 2.0], &device);

    let compile = device.mode(ProcessMode::CompileOnly).unwrap();
    core::mem::drop(compile);

    input
        .add_scalar(1.0)
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TestTensor::<1>::from_data([2.0, 3.0], &device).into_data(),
            Tolerance::default(),
        );
    device
        .mode(ProcessMode::CompileAndAutotune)
        .expect("no mode is left open")
        .close()
        .unwrap();
}

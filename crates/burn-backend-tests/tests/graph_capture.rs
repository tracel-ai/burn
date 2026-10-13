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

extern crate alloc;

#[cfg(feature = "cube")]
pub type FloatElem = f32;
#[cfg(feature = "cube")]
#[allow(unused)]
pub type IntElem = i32;

#[cfg(feature = "cube")]
#[path = "common/backend.rs"]
mod backend;

#[cfg(feature = "cube")]
mod cube {
    use super::{FloatElem, backend::*};

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

        let mut graph =
            burn_tensor::capture(&device, || input.clone().mul_scalar(2.0).add_scalar(1.0));

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
}

use burn_tensor::{CaptureError, Device, Tensor};

#[test]
fn test_capture_scope_boundary_helpers_single_io() {
    let device = Device::capture();

    let captured = device
        .capture_scope(|mut scope| {
            let x = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
            let x = scope.input(x);
            let y = x * 2.0;
            scope.complete_with([&y])
        })
        .expect("Capture should succeed");

    assert_eq!(captured.graph.inputs.len(), 1);
    assert_eq!(captured.graph.outputs.len(), 1);
}

#[test]
fn test_capture_scope_boundary_helpers_tuple_multi_io() {
    let device = Device::capture();

    let mut x1_id = None;
    let mut x2_id = None;

    let captured = device
        .capture_scope(|mut scope| {
            let x1 = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
            let x2 = Tensor::<2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);
            let x1 = scope.input(x1);
            let x2 = scope.input(x2);
            x1_id = Some(x1.capture_id());
            x2_id = Some(x2.capture_id());
            let y1 = x1.clone() + x2.clone();
            let y2 = x1 * 3.0;
            scope.complete_with((&y1, &y2))
        })
        .expect("Capture should succeed");

    assert_eq!(captured.graph.inputs, vec![x1_id.unwrap(), x2_id.unwrap()]);
    assert_eq!(captured.graph.outputs.len(), 2);
}

#[test]
fn test_capture_scope_inner_tensor_as_input_fails() {
    let device = Device::capture();

    let result = device.capture_scope(|mut scope| {
        // Tensor instantiated directly inside capture scope via zeros -> recorded as operation output
        let inner = Tensor::<2>::zeros([2, 2], &device);
        let inner = scope.input(inner);
        let y = inner * 2.0;
        scope.complete_with([&y])
    });

    match result {
        Err(CaptureError::InvalidInput { .. }) => (),
        other => panic!("Expected Err(CaptureError::InvalidInput), got {other:?}"),
    }
}

#[cfg(feature = "flex")]
mod flex_capture_tests {
    use super::*;

    #[test]
    fn test_capture_scope_boundary_helpers_flex_transfer() {
        let capture_device = Device::capture();
        let flex_device = Device::flex();

        let x = Tensor::<2>::zeros([2, 2], &flex_device);

        let captured = capture_device
            .capture_scope(|mut scope| {
                let x = scope.input(x.to_device(&capture_device));
                let y = x * 2.0;
                scope.complete_with([&y])
            })
            .expect("Capture should succeed");

        assert_eq!(captured.graph.inputs.len(), 1);
        assert_eq!(captured.graph.outputs.len(), 1);
    }

    #[test]
    fn test_capture_scope_boundary_helpers_flex_multi_io() {
        let capture_device = Device::capture();
        let flex_device = Device::flex();

        let x1 = Tensor::<2>::zeros([2, 2], &flex_device);
        let x2 = Tensor::<2>::ones([2, 2], &flex_device);

        let mut x1_id = None;
        let mut x2_id = None;

        let captured = capture_device
            .capture_scope(|mut scope| {
                let x1 = scope.input(x1.to_device(&capture_device));
                let x2 = scope.input(x2.to_device(&capture_device));
                x1_id = Some(x1.capture_id());
                x2_id = Some(x2.capture_id());
                let y1 = x1.clone() + x2.clone();
                let y2 = x1 * 3.0;
                scope.complete_with((&y1, &y2))
            })
            .expect("Capture should succeed");

        assert_eq!(captured.graph.inputs, vec![x1_id.unwrap(), x2_id.unwrap()]);
        assert_eq!(captured.graph.outputs.len(), 2);
    }
}

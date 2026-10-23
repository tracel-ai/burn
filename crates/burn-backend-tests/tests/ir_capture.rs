//! Tests for capturing operation graphs and declaring their inputs and outputs.

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

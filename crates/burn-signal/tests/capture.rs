#![cfg(feature = "capture")]

use burn_core::{
    backend::{
        Capture,
        ir::{OperationIr, ScalarIr},
    },
    tensor::{Device, Tensor},
};
use burn_signal::{irfft, rfft};

#[test]
fn capture_records_fft_shapes_outputs_and_arguments() {
    let device = Device::capture();
    let captured = device
        .capture_scope(|scope| {
            let signal = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
            let (real, imag) = rfft(signal, 1, Some(8));
            assert_eq!(real.dims(), [1, 5]);
            assert_eq!(imag.dims(), [1, 5]);
            let output = irfft(real, imag, 1, Some(8));
            assert_eq!(output.dims(), [1, 8]);
            let id = output.try_into_primitive::<Capture>().unwrap().id();
            scope.complete([], [id])
        })
        .unwrap();
    let custom: Vec<_> = captured
        .graph
        .operations
        .iter()
        .filter_map(|op| match op {
            OperationIr::Custom(desc) => Some(desc),
            _ => None,
        })
        .collect();
    assert_eq!(custom.len(), 2);
    assert_eq!(custom[0].id, "signal::rfft");
    assert_eq!(custom[0].outputs.len(), 2);
    assert_eq!(custom[1].id, "signal::irfft");
    assert_eq!(custom[1].inputs.len(), 2);
    assert_eq!(
        custom[0].scalars,
        vec![ScalarIr::UInt(1), ScalarIr::Bool(true), ScalarIr::UInt(8)]
    );
    assert_eq!(custom[0].scalars, custom[1].scalars);
}

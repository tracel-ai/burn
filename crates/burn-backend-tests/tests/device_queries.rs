//! Tests of what a device answers about itself: its description, its capabilities, and its
//! measurements. Run with the `remote` feature, the server answers them.

#![cfg(any(feature = "cube", feature = "remote"))]

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;

use backend::*;
use burn_tensor::{Device, MatmulTile, MemoryAccess, PeakKey};

/// A device on a runtime with a driver describes itself without being opened.
#[test]
fn a_device_describes_itself() {
    let device = Device::default();

    let description = device
        .description()
        .expect("the driver describes the device");

    assert!(!description.name.is_empty(), "{description:?}");
}

/// The counters answer, whichever of them the card keeps: a figure is a reading or nothing,
/// never a failure.
#[test]
fn a_telemetry_reading_answers() {
    let device = Device::default();

    let telemetry = device.measure_telemetry();

    if let Some(busy) = telemetry.busy_percent {
        assert!((0.0..=100.0).contains(&busy), "{telemetry:?}");
    }
}

/// What the allocator holds grows by a tensor kept alive, and the report holds its pools.
#[test]
fn the_memory_report_holds_a_live_tensor() {
    let device = Device::default();
    device.sync().unwrap();
    let before = device
        .measure_memory()
        .expect("the allocator reports")
        .usage;

    let kept = TestTensor::<2>::ones([256, 256], &device);
    device.sync().unwrap();
    let after = device
        .measure_memory()
        .expect("the allocator reports")
        .usage;

    assert!(
        after.bytes_in_use >= before.bytes_in_use + 256 * 256 * 4,
        "{before:?} then {after:?}"
    );
    core::mem::drop(kept);
}

/// A cubecl device states what it can do: who it is, and what its kernels compile for. A
/// remote device states its server's device, without the card.
#[cfg(feature = "cube")]
#[test]
fn a_cubecl_device_states_its_capabilities() {
    let device = Device::default();

    let capabilities = device
        .capabilities()
        .expect("a cubecl device has capabilities");

    assert!(!capabilities.properties.identity.fingerprint.is_empty());
    #[cfg(feature = "remote")]
    if let burn_dispatch::DispatchDevice::Remote(_) = device.as_dispatch() {
        assert_eq!(capabilities.properties.identity.physical, None);
    }
}

/// Every key is answered, in order: a peak, or why there is none — a matrix tile no hardware
/// implements is unsupported rather than a failure of the others.
#[test]
#[ignore = "launches throughput probes, which saturate the device"]
fn every_peak_key_is_answered_in_order() {
    let device = Device::default();
    let keys = [
        PeakKey::Launch,
        PeakKey::Memory(MemoryAccess::Read),
        PeakKey::Matmul {
            dtype: burn_tensor::DType::F16,
            accumulator: burn_tensor::DType::F32,
            tile: MatmulTile { m: 3, n: 5, k: 7 },
        },
    ];

    let peaks = device.measure_peaks(&keys);

    assert_eq!(peaks.len(), keys.len());
    assert!(peaks[2].is_err(), "{peaks:?}");
}

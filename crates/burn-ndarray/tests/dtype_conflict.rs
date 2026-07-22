//! Regression test for <https://github.com/tracel-ai/burn/issues/5192>.
//!
//! The device settings registry is keyed by physical device, so `NdArray<f32>` and `NdArray<f64>`
//! share the same entry. The first backend to touch a device locks its default element width. A
//! later access by a backend with a different width used to silently read back the wrong dtype;
//! it must now fail loudly instead.

use burn_backend::{FloatDType, get_device_settings};
use burn_ndarray::{NdArray, NdArrayDevice};

#[test]
fn conflicting_float_width_on_same_device_fails_loudly() {
    let device = NdArrayDevice::Cpu;

    // First backend locks the device to F32 (this is what the first tensor operation does).
    let settings = get_device_settings::<NdArray<f32>>(&device);
    assert_eq!(settings.float_dtype, FloatDType::F32);

    // Accessing the same device through a backend with a different element width must panic with a
    // clear message rather than silently returning the F32 settings.
    let result = std::panic::catch_unwind(|| get_device_settings::<NdArray<f64>>(&device));
    let payload = result.expect_err("expected a panic on conflicting element width");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        message.contains("Conflicting default data types"),
        "unexpected panic message: {message}"
    );
}

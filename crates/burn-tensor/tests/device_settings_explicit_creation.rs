//! A separate test binary starts with uninitialized device settings.

use burn_tensor::{DType, Device, DeviceError, FloatDType, Tensor};

#[test]
fn explicit_tensor_creation_locks_default_settings() {
    let mut device = Device::default();
    let tensor = Tensor::<1>::from_data([1.0f64], (&device, DType::F64));

    assert_eq!(tensor.dtype(), DType::F64);
    assert_eq!(device.settings().float_dtype, FloatDType::F32);
    assert!(matches!(
        device.configure(FloatDType::F64),
        Err(DeviceError::AlreadyInitialized { .. })
    ));
    assert_eq!(Tensor::<1>::zeros([1], &device).dtype(), DType::F32);
}

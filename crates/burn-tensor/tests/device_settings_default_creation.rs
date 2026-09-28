//! A separate test binary starts with uninitialized device settings.

use burn_tensor::{DType, Device, DeviceError, FloatDType, Tensor};

#[test]
fn tensor_creation_locks_default_settings() {
    let mut device = Device::default();
    let tensor = Tensor::<1>::zeros([1], &device);

    assert_eq!(tensor.dtype(), DType::F32);
    assert!(matches!(
        device.configure(FloatDType::F64),
        Err(DeviceError::AlreadyInitialized { .. })
    ));
}

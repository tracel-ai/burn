//! A separate test binary starts with uninitialized device settings.

use burn_dispatch::devices::FlexDevice;
use burn_tensor::{DType, Device, DeviceError, FloatDType, Tensor};

#[test]
fn tensor_creation_locks_default_settings() {
    // Flex supports F64 regardless of which other backends are enabled.
    let mut device = Device::new(FlexDevice);
    let tensor = Tensor::<1>::zeros([1], &device);

    assert_eq!(tensor.dtype(), DType::F32);
    assert!(matches!(
        device.configure(FloatDType::F64),
        Err(DeviceError::AlreadyInitialized { .. })
    ));
}

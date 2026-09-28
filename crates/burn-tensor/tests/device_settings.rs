//! A separate test binary keeps device configuration independent of other tensor tests.

use burn_tensor::{DType, Device, DeviceError, FloatDType, Tensor};

#[test]
fn querying_settings_allows_configuration_and_observes_it_across_threads() {
    let device = Device::default();
    assert_eq!(device.settings().float_dtype, FloatDType::F32);
    assert_eq!(device.settings().float_dtype, FloatDType::F32);

    let mut other = device.clone();
    std::thread::spawn(move || other.configure(FloatDType::F64).unwrap())
        .join()
        .unwrap();

    assert_eq!(device.settings().float_dtype, FloatDType::F64);
    let tensor = Tensor::<1>::zeros([1], &device);
    assert_eq!(tensor.dtype(), DType::F64);

    let mut device = device;
    assert!(matches!(
        device.configure(FloatDType::F32),
        Err(DeviceError::AlreadyInitialized { .. })
    ));
}

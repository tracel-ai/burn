pub use burn_std::device::*;
use burn_std::{BoolDType, DType, FloatDType, IntDType};

use alloc::format;
use alloc::string::String;
use burn_std::stub::RwLock;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;
use thiserror::Error;

use core::any::TypeId;

#[cfg(feature = "std")]
pub use std::collections::HashMap;
#[cfg(feature = "std")]
use std::sync::LazyLock;

#[cfg(not(feature = "std"))]
pub use hashbrown::HashMap;
#[cfg(not(feature = "std"))]
use spin::Lazy as LazyLock;

use crate::Backend;

/// Device trait for all burn backend devices.
pub trait DeviceOps: Clone + Default + PartialEq + Send + Sync + core::fmt::Debug + Device {
    /// Returns the [device id](DeviceId).
    fn id(&self) -> DeviceId {
        self.to_id()
    }

    /// Returns the inner device without autodiff enabled.
    ///
    /// For most devices this is a no-op that returns `self`. For autodiff-enabled
    /// devices, this returns the underlying inner device.
    fn inner(&self) -> &Self {
        self
    }
}

/// Settings controlling the default device data types.
///
/// These settings follow snapshot semantics. When you retrieve settings for a device,
/// you receive an immutable snapshot of the current configuration. Changes made to the global
/// [DeviceSettings] registry will only affect tensors created after the policy
/// was updated.
///
/// Settings should only be set once during initialization, to be queried during tensor creation.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceSettings {
    /// Default floating-point data type for tensor creation.
    float_dtype: Option<FloatDType>,
    /// Default integer data type for tensor creation.
    int_dtype: Option<IntDType>,
    /// Default bool data type for tensor creation.
    bool_dtype: Option<BoolDType>,
}

impl DeviceSettings {
    /// Returns the default floating-point data type.
    ///
    /// The dtype is resolved from the device settings if set, otherwise falls back
    /// to the backend default.
    pub fn float_dtype<B: Backend>(&self) -> FloatDType {
        self.float_dtype
            .unwrap_or(<B::FloatElem as crate::Element>::dtype().into())
    }

    /// Returns the default integer data type.
    ///
    /// The dtype is resolved from the device settings if set, otherwise falls back
    /// to the backend default.
    pub fn int_dtype<B: Backend>(&self) -> IntDType {
        self.int_dtype
            .unwrap_or(<B::IntElem as crate::Element>::dtype().into())
    }

    /// Returns the bool data type for the given device.
    ///
    /// The dtype is resolved from the device settings if set, otherwise falls back
    /// to the backend default.
    pub fn bool_dtype<B: Backend>(&self) -> BoolDType {
        // TODO: auto configure for Dispatch backend otherwise won't have correct dtype
        self.bool_dtype
            .unwrap_or(<B::BoolElem as crate::Element>::dtype().into())
    }

    /// Sets the default floating-point data type.
    pub(crate) fn set_float_dtype(&mut self, dtype: FloatDType) {
        self.float_dtype = Some(dtype);
    }

    /// Sets the default integer data type.
    pub(crate) fn set_int_dtype(&mut self, dtype: IntDType) {
        self.int_dtype = Some(dtype);
    }
}

/// Key for the registry: physical device type + device id
type RegistryKey = (DeviceId, TypeId);

// TODO: use OnceLock<DeviceSettings> to enforce the "initialized once" contract.

/// Global registry mapping devices to their settings.
static REGISTRY: LazyLock<RwLock<HashMap<RegistryKey, Arc<DeviceSettings>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Device settings management for controlling default tensor creation behavior.
///
/// # Settings Semantics
///
/// Device settings use snapshot semantics: when you retrieve a settings with
/// [`get_device_settings`], you get an immutable snapshot of the current configuration.
/// Updates to the settings (via [`set_default_dtypes`], [`set_default_float_dtype`], etc.)
/// only affect future settings retrievals, not existing references.
///
/// This is intended for the common case where settings are set once during
/// initialization and then read frequently during tensor creation.
struct DeviceSettingsRegistry;

impl DeviceSettingsRegistry {
    /// Get the settings for a physical device type and device id.
    ///
    /// If no settings exists yet, a default one is created and stored.
    fn get<D: DeviceOps>(device: &D) -> Arc<DeviceSettings> {
        let key = Self::key(device);

        if let Some(settings) = REGISTRY.read().unwrap().get(&key) {
            return Arc::clone(settings);
        }

        let mut map = REGISTRY.write().unwrap();
        Arc::clone(
            map.entry(key)
                .or_insert_with(|| Arc::new(DeviceSettings::default())),
        )
    }

    /// Mutate the settings for a given device.
    fn update<D: DeviceOps>(device: &D, update_fn: impl FnOnce(&mut DeviceSettings)) {
        let key = Self::key(device);
        let mut map = REGISTRY.write().unwrap();

        let settings = map
            .entry(key)
            .or_insert_with(|| Arc::new(DeviceSettings::default()));

        // Update the settings
        let settings_mut = Arc::make_mut(settings);
        update_fn(settings_mut);
    }

    /// Returns the device registry key.
    fn key<D: Device>(device: &D) -> RegistryKey {
        (device.to_id(), TypeId::of::<D>())
    }
}

/// Get the [`device`'s settings](DeviceSettings).
///
/// Returns an immutable snapshot of the device's current settings. If the settings
/// is updated after retrieval, this snapshot will not reflect those changes.
pub fn get_device_settings<D: DeviceOps>(device: &D) -> Arc<DeviceSettings> {
    DeviceSettingsRegistry::get(device)
}

/// Errors that can occur during device-related operations.
///
/// This covers errors related to hardware capability mismatches, such as
/// requesting a data type not supported by the device, and configuration
/// errors like attempting to change a settings in an invalid context.
#[derive(Debug, Error)]
pub enum DeviceError {
    /// Unsupported data type by the device.
    #[error("Device {device} does not support the requested data type {dtype:?}")]
    UnsupportedDType {
        /// The string representation of the device.
        device: String,
        /// The data type that caused the error.
        dtype: DType,
    },
    // TODO: `InvalidContext` if a device settings cannot be changed after init / during training / etc.
}

impl DeviceError {
    /// Helper to create a [`DeviceError::UnsupportedDType`] from any device.
    pub fn unsupported_dtype<D: DeviceOps>(device: &D, dtype: DType) -> Self {
        Self::UnsupportedDType {
            device: format!("{device:?}"),
            dtype,
        }
    }
}

fn check_dtype_support<B: Backend>(
    device: &B::Device,
    dtype: impl Into<DType>,
) -> Result<(), DeviceError> {
    let dtype = dtype.into();
    // Default dtypes should have `DTypeUsage::general()`. Types restricted to specialized
    // operations should not be used as default.
    if B::supports_dtype(device, dtype) {
        Ok(())
    } else {
        Err(DeviceError::unsupported_dtype(device, dtype))
    }
}

/// Sets the default data types for the device.
///
/// This updates the device's default data types used for tensor creation.
/// The settings should typically be set once during initialization and then
/// remains global for all subsequent operations on that device.
///
/// # Example
///
/// ```rust
/// use burn_tensor::backend::Backend;
/// use burn_tensor::{DType, Int, Tensor, set_default_dtypes};
///
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     
///     // Update the device settings
///     set_default_dtypes::<B>(&device, DType::F16, DType::I32);
///     
///     // All float tensors created after this will use F16 by default
///     let tensor = Tensor::<B, 2>::zeros([2, 3], &device);
///     // All int tensors created after this will use I32 default
///     let tensor = Tensor::<B, 2, Int>::zeros([2, 3], &device);
/// }
/// ```
pub fn set_default_dtypes<B: Backend>(
    device: &B::Device,
    float_dtype: impl Into<FloatDType>,
    int_dtype: impl Into<IntDType>,
) -> Result<(), DeviceError> {
    let float_dtype = float_dtype.into();
    let int_dtype = int_dtype.into();
    check_dtype_support::<B>(device, float_dtype)?;
    check_dtype_support::<B>(device, int_dtype)?;

    set_default_dtypes_unchecked(device, float_dtype, int_dtype);
    Ok(())
}

/// Sets the default floating-point data type for the device.
///
/// This updates the device's default data types used for tensor creation.
/// The settings should typically be set once during initialization and then
/// remains global for all subsequent operations on that device.
///
/// # Example
///
/// ```rust
/// use burn_tensor::backend::Backend;
/// use burn_tensor::{DType, Tensor, set_default_float_dtype};
///
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     
///     // Update the device settings
///     set_default_float_dtype::<B>(&device, DType::F16);
///     
///     // All float tensors created after this will use F16 by default
///     let tensor = Tensor::<B, 2>::zeros([2, 3], &device);
/// }
/// ```
pub fn set_default_float_dtype<B: Backend>(
    device: &B::Device,
    dtype: impl Into<FloatDType>,
) -> Result<(), DeviceError> {
    let dtype = dtype.into();
    check_dtype_support::<B>(device, dtype)?;

    set_default_float_dtype_unchecked(device, dtype);
    Ok(())
}

/// Sets the default integer data type for the device.
///
/// This updates the device's default data types used for tensor creation.
/// The settings should typically be set once during initialization and then
/// remains global for all subsequent operations on that device.
///
/// # Example
///
/// ```rust
/// use burn_tensor::backend::Backend;
/// use burn_tensor::{DType, Int, Tensor, set_default_int_dtype};
///
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     
///     // Update the device settings
///     set_default_int_dtype::<B>(&device, DType::I32);
///     
///     // All int tensors created after this will use I32 default
///     let tensor = Tensor::<B, 2, Int>::zeros([2, 3], &device);
/// }
/// ```
pub fn set_default_int_dtype<B: Backend>(
    device: &B::Device,
    dtype: impl Into<IntDType>,
) -> Result<(), DeviceError> {
    let dtype = dtype.into();
    check_dtype_support::<B>(device, dtype)?;

    set_default_int_dtype_unchecked(device, dtype);
    Ok(())
}

// Unchecked versions
fn set_default_dtypes_unchecked<D: DeviceOps>(
    device: &D,
    float_dtype: FloatDType,
    int_dtype: IntDType,
) {
    DeviceSettingsRegistry::update(device, |p| {
        p.set_float_dtype(float_dtype);
        p.set_int_dtype(int_dtype);
    });
}

fn set_default_float_dtype_unchecked<D: DeviceOps>(device: &D, dtype: FloatDType) {
    DeviceSettingsRegistry::update(device, |p| {
        p.set_float_dtype(dtype);
    });
}

fn set_default_int_dtype_unchecked<D: DeviceOps>(device: &D, dtype: IntDType) {
    DeviceSettingsRegistry::update(device, |p| {
        p.set_int_dtype(dtype);
    });
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use serial_test::serial;

    use super::*;

    fn clear_registry() {
        REGISTRY.write().unwrap().clear();
    }

    #[derive(Clone, Debug, Default, PartialEq, new)]
    pub struct TestDeviceA {
        index: u32,
    }

    impl Device for TestDeviceA {
        fn from_id(device_id: DeviceId) -> Self {
            Self {
                index: device_id.index_id,
            }
        }

        fn to_id(&self) -> DeviceId {
            DeviceId {
                type_id: 0,
                index_id: self.index,
            }
        }

        fn device_count(_type_id: u16) -> usize {
            1
        }
    }

    impl DeviceOps for TestDeviceA {}

    #[derive(Clone, Debug, Default, PartialEq, new)]
    pub struct TestDeviceB {
        index: u32,
    }

    impl Device for TestDeviceB {
        fn from_id(device_id: DeviceId) -> Self {
            Self {
                index: device_id.index_id,
            }
        }

        fn to_id(&self) -> DeviceId {
            DeviceId {
                type_id: 0,
                index_id: self.index,
            }
        }

        fn device_count(_type_id: u16) -> usize {
            1
        }
    }

    impl DeviceOps for TestDeviceB {}

    #[test]
    #[serial]
    fn default_settings_is_created_and_shared() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        let p1 = get_device_settings(&device);
        let p2 = get_device_settings(&device);

        assert!(Arc::ptr_eq(&p1, &p2));
        // Not explicitly set
        assert!(p1.float_dtype.is_none());
        assert!(p1.int_dtype.is_none());
        assert!(p2.float_dtype.is_none());
        assert!(p2.int_dtype.is_none());
    }

    #[test]
    #[serial]
    fn updated_settings_is_shared() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        // The device settings is meant to be set once at initialization
        set_default_dtypes_unchecked(&device, FloatDType::BF16, IntDType::I32);
        let p1 = get_device_settings(&device);
        let p2 = get_device_settings(&device);

        assert!(Arc::ptr_eq(&p1, &p2));
        assert_eq!(p1.float_dtype, Some(FloatDType::BF16));
        assert_eq!(p1.int_dtype, Some(IntDType::I32));
        assert_eq!(p2.float_dtype, Some(FloatDType::BF16));
        assert_eq!(p2.int_dtype, Some(IntDType::I32));
    }

    #[test]
    #[serial]
    fn settings_is_device_id_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceA::new(1);

        set_default_float_dtype_unchecked(&d1, FloatDType::F16);

        let p1 = get_device_settings(&d1);
        let p2 = get_device_settings(&d2);

        assert!(!Arc::ptr_eq(&p1, &p2));
        assert_eq!(p1.float_dtype, Some(FloatDType::F16));
        assert!(p1.int_dtype.is_none());
        assert!(p2.float_dtype.is_none());
        assert!(p2.int_dtype.is_none());
    }

    #[test]
    #[serial]
    fn settings_is_device_type_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceB::new(0);

        set_default_float_dtype_unchecked(&d2, FloatDType::F16);

        let p1 = get_device_settings(&d1);
        let p2 = get_device_settings(&d2);

        assert!(p1.float_dtype.is_none());
        assert!(p1.int_dtype.is_none());
        assert_eq!(p2.float_dtype, Some(FloatDType::F16));
        assert!(p2.int_dtype.is_none());
    }

    #[test]
    #[serial]
    fn updating_settings_should_not_affect_snapshot() {
        clear_registry(); // reset registry for each test

        // The device settings is meant to be set once at initialization
        let device = TestDeviceA::new(0);
        let before = get_device_settings(&device);

        set_default_float_dtype_unchecked(&device, FloatDType::BF16);

        let after = get_device_settings(&device);

        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(after.float_dtype, Some(FloatDType::BF16));
        assert!(before.float_dtype.is_none());
    }

    #[test]
    #[serial]
    fn set_default_dtypes_overwrites_fields() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        set_default_dtypes_unchecked(&device, FloatDType::F16, IntDType::I64);

        let settings = get_device_settings(&device);

        assert_eq!(settings.float_dtype, Some(FloatDType::F16));
        assert_eq!(settings.int_dtype, Some(IntDType::I64));
    }
}

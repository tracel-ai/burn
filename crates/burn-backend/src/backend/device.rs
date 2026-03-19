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
use std::sync::{LazyLock, OnceLock};

#[cfg(not(feature = "std"))]
pub use hashbrown::HashMap;
#[cfg(not(feature = "std"))]
use spin::{Lazy as LazyLock, Once as OnceLock};

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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
}

/// Key for the registry: physical device type + device id
type RegistryKey = (DeviceId, TypeId);

/// Global registry mapping devices to their settings.
///
/// Each value is wrapped in a `OnceLock` to enforce that settings are initialized only once
/// per device.
static REGISTRY: LazyLock<RwLock<HashMap<RegistryKey, Arc<OnceLock<DeviceSettings>>>>> =
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
    /// Returns the settings for the given device, inserting the default if absent.
    fn get_or_insert<D: DeviceOps>(device: &D) -> DeviceSettings {
        let key = Self::key(device);
        #[cfg(feature = "std")]
        {
            let cached = LOCAL_CACHE.with(|cache| cache.borrow().get(&key).copied());
            if let Some(settings) = cached {
                return settings;
            }

            // Entry does not exist in cache
            let settings = {
                let read = REGISTRY.read().unwrap();
                read.get(&key).cloned()
            }
            .unwrap_or_else(|| {
                let mut map = REGISTRY.write().unwrap();
                Arc::clone(map.entry(key).or_default())
            });

            let settings = *settings.get_or_init(DeviceSettings::default);

            LOCAL_CACHE.with(|cache| {
                cache.borrow_mut().insert(key, settings);
            });

            settings
        }
        #[cfg(not(feature = "std"))]
        {
            let settings = {
                let read = REGISTRY.read().unwrap();
                read.get(&key).cloned()
            }
            .unwrap_or_else(|| {
                let mut map = REGISTRY.write().unwrap();
                Arc::clone(map.entry(key).or_default())
            });

            settings.call_once(DeviceSettings::default);
            *settings.get().unwrap()
        }
    }

    /// Initializes the settings for the given device.
    ///
    /// Returns `Err` with the existing settings if already initialized.
    fn init<D: DeviceOps>(device: &D, settings: DeviceSettings) -> Result<(), DeviceError> {
        let key = Self::key(device);
        let mut map = REGISTRY.write().unwrap();
        let cell = map.entry(key).or_insert_with(|| Arc::new(OnceLock::new()));

        #[cfg(feature = "std")]
        return cell
            .set(settings)
            .map_err(|_| DeviceError::already_initialized(device));

        #[cfg(not(feature = "std"))]
        if cell.get().is_some() {
            Err(DeviceError::already_initialized(device))
        } else {
            cell.call_once(|| settings);
            Ok(())
        }
    }

    /// Returns the device registry key.
    fn key<D: Device>(device: &D) -> RegistryKey {
        (device.to_id(), TypeId::of::<D>())
    }
}

#[cfg(feature = "std")]
thread_local! {
    /// Thread-local cache access to initialized device settings is lock-free.
    static LOCAL_CACHE: core::cell::RefCell<HashMap<RegistryKey, DeviceSettings>> =
        core::cell::RefCell::new(HashMap::new());
}

/// Get the [`device`'s settings](DeviceSettings).
///
/// Returns an immutable snapshot of the device's current settings. If the settings
/// is updated after retrieval, this snapshot will not reflect those changes.
pub fn get_device_settings<D: DeviceOps>(device: &D) -> DeviceSettings {
    DeviceSettingsRegistry::get_or_insert(device)
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
    /// Device settings have already been initialized.
    #[error("Device {device} settings have already been initialized")]
    AlreadyInitialized {
        /// The string representation of the device.
        device: String,
    },
}

impl DeviceError {
    /// Helper to create a [`DeviceError::UnsupportedDType`] from any device.
    pub fn unsupported_dtype<D: DeviceOps>(device: &D, dtype: DType) -> Self {
        Self::UnsupportedDType {
            device: format!("{device:?}"),
            dtype,
        }
    }

    /// Helper to create a [`DeviceError::AlreadyInitialized`] from any device.
    pub fn already_initialized<D: DeviceOps>(device: &D) -> Self {
        Self::AlreadyInitialized {
            device: format!("{device:?}"),
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
///
/// Settings can only be initialized once per device. Subsequent calls for
/// the same device return [`DeviceError::AlreadyInitialized`].
///
/// # Note
///
/// Initialization must happen before any tensor creation on the device.
/// The first tensor operation will lock the device to its defaults, causing
/// any subsequent initialization attempt to return [`DeviceError::AlreadyInitialized`].
///
/// # Example
///
/// ```rust, ignore
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

    initialize_unchecked(device, Some(float_dtype), Some(int_dtype), None)?;
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
/// ```rust, ignore
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

    initialize_unchecked(device, Some(dtype), None, None)?;
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
/// ```rust, ignore
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

    initialize_unchecked(device, None, Some(dtype), None)?;
    Ok(())
}

// Unchecked dtypes
fn initialize_unchecked<D: DeviceOps>(
    device: &D,
    float_dtype: Option<FloatDType>,
    int_dtype: Option<IntDType>,
    bool_dtype: Option<BoolDType>,
) -> Result<(), DeviceError> {
    DeviceSettingsRegistry::init(
        device,
        DeviceSettings {
            float_dtype,
            int_dtype,
            bool_dtype,
        },
    )
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
    fn default_settings_returned_when_uninitialized() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        let s1 = get_device_settings(&device);
        let s2 = get_device_settings(&device);

        assert_eq!(s1, s2);
        // Not explicitly set
        assert!(s1.float_dtype.is_none());
        assert!(s1.int_dtype.is_none());
        assert!(s2.float_dtype.is_none());
        assert!(s2.int_dtype.is_none());
    }

    #[test]
    #[serial]
    fn initialized_settings_are_returned() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        initialize_unchecked(&device, Some(FloatDType::BF16), Some(IntDType::I32), None).unwrap();
        let s1 = get_device_settings(&device);
        let s2 = get_device_settings(&device);

        assert_eq!(s1, s2);
        assert_eq!(s1.float_dtype, Some(FloatDType::BF16));
        assert_eq!(s1.int_dtype, Some(IntDType::I32));
        assert_eq!(s2.float_dtype, Some(FloatDType::BF16));
        assert_eq!(s2.int_dtype, Some(IntDType::I32));
    }

    #[test]
    #[serial]
    fn settings_are_device_id_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceA::new(1);

        initialize_unchecked(&d1, Some(FloatDType::F16), None, None).unwrap();

        let s1 = get_device_settings(&d1);
        let s2 = get_device_settings(&d2);

        assert_ne!(s1, s2);
        assert_eq!(s1.float_dtype, Some(FloatDType::F16));
        assert!(s1.int_dtype.is_none());
        assert!(s2.float_dtype.is_none());
        assert!(s2.int_dtype.is_none());
    }

    #[test]
    #[serial]
    fn settings_are_device_type_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceB::new(0);

        initialize_unchecked(&d2, Some(FloatDType::F16), None, None).unwrap();

        let s1 = get_device_settings(&d1);
        let s2 = get_device_settings(&d2);

        assert!(s1.float_dtype.is_none());
        assert!(s1.int_dtype.is_none());
        assert_eq!(s2.float_dtype, Some(FloatDType::F16));
        assert!(s2.int_dtype.is_none());
    }

    #[test]
    #[serial]
    fn initialization_after_default_returns_error() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);
        // Settings are set to default on first access, which forces consistency
        let _before = get_device_settings(&device);

        let result = initialize_unchecked(&device, Some(FloatDType::BF16), None, None);

        assert!(matches!(
            result,
            Err(DeviceError::AlreadyInitialized { .. })
        ));
    }

    #[test]
    #[serial]
    fn second_initialization_returns_error() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);
        initialize_unchecked(&device, Some(FloatDType::F16), Some(IntDType::I32), None).unwrap();

        let result =
            initialize_unchecked(&device, Some(FloatDType::BF16), Some(IntDType::I64), None);
        assert!(matches!(
            result,
            Err(DeviceError::AlreadyInitialized { .. })
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    #[serial]
    fn initialized_settings_are_global() {
        clear_registry();

        let device = TestDeviceA::new(0);

        initialize_unchecked(&device, Some(FloatDType::F16), Some(IntDType::I32), None).unwrap();
        let settings = get_device_settings(&device);
        assert_eq!(settings.float_dtype, Some(FloatDType::F16));
        assert_eq!(settings.int_dtype, Some(IntDType::I32));
        assert_eq!(settings.bool_dtype, None);

        // The other thread will see the initialized settings
        let seen_by_new_thread =
            std::thread::spawn(move || get_device_settings(&TestDeviceA::new(0)))
                .join()
                .unwrap();
        assert_eq!(seen_by_new_thread, settings);
    }
}

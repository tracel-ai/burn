pub use burn_std::device::*;
use burn_std::{BoolDType, DType, FloatDType, IntDType};
pub use burn_std::{DeviceError, DeviceSettings};

use burn_std::stub::RwLock;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

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

    /// Returns the default [settings](DeviceSettings) for this device.
    fn defaults(&self) -> DeviceSettings;
}

/// Key for the registry: physical device type + device id
type RegistryKey = (DeviceId, TypeId);

/// Global registry mapping devices to their settings.
///
/// Each value is wrapped in a `OnceLock` to enforce that settings are initialized only once
/// per device.
static REGISTRY: LazyLock<RwLock<HashMap<RegistryKey, Arc<OnceLock<DeviceSettings>>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

struct DeviceSettingsRegistry;

impl DeviceSettingsRegistry {
    /// Returns the settings for the given device, inserting the default if absent.
    fn get_or_insert<D: DeviceOps>(
        device: &D,
        default_fn: impl FnOnce() -> DeviceSettings,
    ) -> DeviceSettings {
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

            let settings = *settings.get_or_init(default_fn);

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

            settings.call_once(default_fn);
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
pub fn get_device_settings<B: Backend>(device: &B::Device) -> DeviceSettings {
    DeviceSettingsRegistry::get_or_insert(device, || device.defaults())
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
    bool_dtype: impl Into<BoolDType>,
) -> Result<(), DeviceError> {
    let float_dtype = float_dtype.into();
    let int_dtype = int_dtype.into();
    let bool_dtype = bool_dtype.into();
    check_dtype_support::<B>(device, float_dtype)?;
    check_dtype_support::<B>(device, int_dtype)?;
    check_dtype_support::<B>(device, bool_dtype)?;

    let q_config = device.defaults().quantization;
    let settings = DeviceSettings::new(float_dtype, int_dtype, bool_dtype, q_config);

    initialize_unchecked(device, settings)?;
    Ok(())
}

// Unchecked dtypes
fn initialize_unchecked<D: DeviceOps>(
    device: &D,
    settings: DeviceSettings,
) -> Result<(), DeviceError> {
    DeviceSettingsRegistry::init(device, settings)
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
        index: u16,
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
    }

    impl DeviceOps for TestDeviceA {
        fn defaults(&self) -> DeviceSettings {
            DeviceSettings::with_dtypes(FloatDType::F32, IntDType::I32, BoolDType::Native)
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, new)]
    pub struct TestDeviceB {
        index: u16,
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
    }

    impl DeviceOps for TestDeviceB {
        fn defaults(&self) -> DeviceSettings {
            DeviceSettings::with_dtypes(FloatDType::F32, IntDType::I32, BoolDType::Native)
        }
    }

    fn get_test_device_settings<D: DeviceOps>(device: &D) -> DeviceSettings {
        DeviceSettingsRegistry::get_or_insert(device, || device.defaults())
    }

    #[test]
    #[serial]
    fn default_settings_returned_when_uninitialized() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        let s1 = get_test_device_settings(&device);
        let s2 = get_test_device_settings(&device);

        assert_eq!(s1, s2);
        assert_eq!(s1, device.defaults());
    }

    #[test]
    #[serial]
    fn initialized_settings_are_returned() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);
        let settings =
            DeviceSettings::with_dtypes(FloatDType::BF16, IntDType::I32, BoolDType::Native);

        initialize_unchecked(&device, settings).unwrap();
        let s1 = get_test_device_settings(&device);
        let s2 = get_test_device_settings(&device);

        assert_eq!(s1, s2);
        assert_eq!(s1, settings);
        assert_eq!(s2, settings);
    }

    #[test]
    #[serial]
    fn settings_are_device_id_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceA::new(1);
        let settings =
            DeviceSettings::with_dtypes(FloatDType::F16, IntDType::I64, BoolDType::Native);

        initialize_unchecked(&d1, settings).unwrap();

        let s1 = get_test_device_settings(&d1);
        let s2 = get_test_device_settings(&d2);

        assert_ne!(s1, s2);
        assert_eq!(s1, settings);
        assert_eq!(s2, d2.defaults());
    }

    #[test]
    #[serial]
    fn settings_are_device_type_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceB::new(0);
        let settings =
            DeviceSettings::with_dtypes(FloatDType::F16, IntDType::I64, BoolDType::Native);

        initialize_unchecked(&d2, settings).unwrap();

        let s1 = get_test_device_settings(&d1);
        let s2 = get_test_device_settings(&d2);

        assert_ne!(s1, s2);
        assert_eq!(s1, d1.defaults());
        assert_eq!(s2, settings);
    }

    #[test]
    #[serial]
    fn initialization_after_default_returns_error() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);
        // Settings are set to default on first access, which forces consistency
        let _before = get_test_device_settings(&device);

        let settings =
            DeviceSettings::with_dtypes(FloatDType::BF16, IntDType::I64, BoolDType::Native);
        let result = initialize_unchecked(&device, settings);

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
        let settings =
            DeviceSettings::with_dtypes(FloatDType::F16, IntDType::I32, BoolDType::Native);
        initialize_unchecked(&device, settings).unwrap();

        let result = initialize_unchecked(&device, device.defaults());
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
        let settings =
            DeviceSettings::with_dtypes(FloatDType::F16, IntDType::I32, BoolDType::Native);

        initialize_unchecked(&device, settings).unwrap();
        let settings_actual = get_test_device_settings(&device);
        assert_eq!(settings_actual, settings);

        // The other thread will see the initialized settings
        let seen_by_new_thread =
            std::thread::spawn(move || get_test_device_settings(&TestDeviceA::new(0)))
                .join()
                .unwrap();
        assert_eq!(seen_by_new_thread, settings_actual);
    }
}

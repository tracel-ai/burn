use alloc::format;
use alloc::string::String;
use burn_backend::{Backend, Device, DeviceId, DeviceOps};
use burn_std::stub::RwLock;
use burn_std::{DType, FloatDType, IntDType};

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

/// Policy controlling default device behavior.
///
/// This includes default data types used for tensor creation.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DevicePolicy {
    /// Default floating-point data type for tensor creation.
    float_dtype: Option<FloatDType>,
    /// Default integer data type for tensor creation.
    int_dtype: Option<IntDType>,
}

impl DevicePolicy {
    /// Returns the default floating-point data type used for tensor creation.
    pub(crate) fn float_dtype(&self) -> Option<FloatDType> {
        self.float_dtype
    }

    /// Returns the default integer data type used for tensor creation.
    pub(crate) fn int_dtype(&self) -> Option<IntDType> {
        self.int_dtype
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

/// Global registry mapping devices to their policies.
static REGISTRY: LazyLock<RwLock<HashMap<RegistryKey, Arc<DevicePolicy>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Device policy management for controlling default tensor creation behavior.
///
/// # Policy Semantics
///
/// Device policies use snapshot semantics: when you retrieve a policy with
/// [`get_device_policy`], you get an immutable snapshot of the current configuration.
/// Updates to the policy (via [`set_default_dtypes`], [`set_default_float_dtype`], etc.)
/// only affect future policy retrievals, not existing references.
///
/// This is intended for the common case where policies are set once during
/// initialization and then read frequently during tensor creation.
struct DevicePolicyRegistry;

impl DevicePolicyRegistry {
    /// Get the policy for a physical device type and device id.
    ///
    /// If no policy exists yet, a default one is created and stored.
    fn get<D: DeviceOps>(device: &D) -> Arc<DevicePolicy> {
        let key = Self::key(device);

        if let Some(policy) = REGISTRY.read().unwrap().get(&key) {
            return Arc::clone(policy);
        }

        let mut map = REGISTRY.write().unwrap();
        Arc::clone(
            map.entry(key)
                .or_insert_with(|| Arc::new(DevicePolicy::default())),
        )
    }

    /// Mutate the policy for a given device.
    fn update<D: DeviceOps>(device: &D, update_fn: impl FnOnce(&mut DevicePolicy)) {
        let key = Self::key(device);
        let mut map = REGISTRY.write().unwrap();

        let policy = map
            .entry(key)
            .or_insert_with(|| Arc::new(DevicePolicy::default()));

        // Update the policy
        let policy_mut = Arc::make_mut(policy);
        update_fn(policy_mut);
    }

    /// Returns the device registry key.
    fn key<D: Device>(device: &D) -> RegistryKey {
        (device.to_id(), TypeId::of::<D>())
    }
}

/// Get the [`device`'s policy](DevicePolicy).
///
/// Returns an immutable snapshot of the device's current policy. If the policy
/// is updated after retrieval, this snapshot will not reflect those changes.
pub(crate) fn get_device_policy<D: DeviceOps>(device: &D) -> Arc<DevicePolicy> {
    DevicePolicyRegistry::get(device)
}

/// Errors that can occur during device-related operations.
///
/// This covers errors related to hardware capability mismatches, such as
/// requesting a data type not supported by the device, and configuration
/// errors like attempting to change a policy in an invalid context.
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
    // TODO: `InvalidContext` if a device policy cannot be changed after init / during training / etc.
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
    if B::supports_dtype(device, dtype) {
        Ok(())
    } else {
        Err(DeviceError::unsupported_dtype(device, dtype))
    }
}

/// Sets the default data types for the device.
///
/// This updates the device's default data types used for tensor creation.
/// The policy should typically be set once during initialization and then
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
///     // Update the device policy
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

    Ok(set_default_dtypes_unchecked(device, float_dtype, int_dtype))
}

/// Sets the default floating-point data type for the device.
///
/// This updates the device's default data types used for tensor creation.
/// The policy should typically be set once during initialization and then
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
///     // Update the device policy
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

    Ok(set_default_float_dtype_unchecked(device, dtype))
}

/// Sets the default integer data type for the device.
///
/// This updates the device's default data types used for tensor creation.
/// The policy should typically be set once during initialization and then
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
///     // Update the device policy
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

    Ok(set_default_int_dtype_unchecked(device, dtype))
}

// Unchecked versions
fn set_default_dtypes_unchecked<D: DeviceOps>(
    device: &D,
    float_dtype: FloatDType,
    int_dtype: IntDType,
) {
    DevicePolicyRegistry::update(device, |p| {
        p.set_float_dtype(float_dtype);
        p.set_int_dtype(int_dtype);
    });
}

fn set_default_float_dtype_unchecked<D: DeviceOps>(device: &D, dtype: FloatDType) {
    DevicePolicyRegistry::update(device, |p| {
        p.set_float_dtype(dtype);
    });
}

fn set_default_int_dtype_unchecked<D: DeviceOps>(device: &D, dtype: IntDType) {
    DevicePolicyRegistry::update(device, |p| {
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
    fn default_policy_is_created_and_shared() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        let p1 = get_device_policy(&device);
        let p2 = get_device_policy(&device);

        assert!(Arc::ptr_eq(&p1, &p2));
        // Not explicitly set
        assert!(p1.float_dtype().is_none());
        assert!(p1.int_dtype().is_none());
        assert!(p2.float_dtype().is_none());
        assert!(p2.int_dtype().is_none());
    }

    #[test]
    #[serial]
    fn updated_policy_is_shared() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        // The device policy is meant to be set once at initialization
        set_default_dtypes_unchecked(&device, FloatDType::BF16, IntDType::I32);
        let p1 = get_device_policy(&device);
        let p2 = get_device_policy(&device);

        assert!(Arc::ptr_eq(&p1, &p2));
        assert_eq!(p1.float_dtype(), Some(FloatDType::BF16));
        assert_eq!(p1.int_dtype(), Some(IntDType::I32));
        assert_eq!(p2.float_dtype(), Some(FloatDType::BF16));
        assert_eq!(p2.int_dtype(), Some(IntDType::I32));
    }

    #[test]
    #[serial]
    fn policy_is_device_id_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceA::new(1);

        set_default_float_dtype_unchecked(&d1, FloatDType::F16);

        let p1 = get_device_policy(&d1);
        let p2 = get_device_policy(&d2);

        assert!(!Arc::ptr_eq(&p1, &p2));
        assert_eq!(p1.float_dtype(), Some(FloatDType::F16));
        assert!(p1.int_dtype().is_none());
        assert!(p2.float_dtype().is_none());
        assert!(p2.int_dtype().is_none());
    }

    #[test]
    #[serial]
    fn policy_is_device_type_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceB::new(0);

        set_default_float_dtype_unchecked(&d2, FloatDType::F16);

        let p1 = get_device_policy(&d1);
        let p2 = get_device_policy(&d2);

        assert!(p1.float_dtype().is_none());
        assert!(p1.int_dtype().is_none());
        assert_eq!(p2.float_dtype(), Some(FloatDType::F16));
        assert!(p2.int_dtype().is_none());
    }

    #[test]
    #[serial]
    fn updating_policy_should_not_affect_snapshot() {
        clear_registry(); // reset registry for each test

        // The device policy is meant to be set once at initialization
        let device = TestDeviceA::new(0);
        let before = get_device_policy(&device);

        set_default_float_dtype_unchecked(&device, FloatDType::BF16);

        let after = get_device_policy(&device);

        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(after.float_dtype(), Some(FloatDType::BF16));
        assert!(before.float_dtype().is_none());
    }

    #[test]
    #[serial]
    fn set_default_dtypes_overwrites_fields() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        set_default_dtypes_unchecked(&device, FloatDType::F16, IntDType::I64);

        let policy = get_device_policy(&device);

        assert_eq!(policy.float_dtype(), Some(FloatDType::F16));
        assert_eq!(policy.int_dtype(), Some(IntDType::I64));
    }
}

use burn_backend::{Backend, Device, DeviceId, DeviceOps, Element};
use burn_std::stub::RwLock;
use burn_std::{FloatDType, IntDType};

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

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
#[derive(Debug, Clone, Copy)]
pub struct DevicePolicy {
    /// Default floating-point data type for tensor creation.
    float_dtype: FloatDType,
    /// Default integer data type for tensor creation.
    int_dtype: IntDType,
}

impl DevicePolicy {
    /// Returns the default floating-point data type used for tensor creation.
    pub fn float_dtype(&self) -> FloatDType {
        self.float_dtype
    }

    /// Returns the default integer data type used for tensor creation.
    pub fn int_dtype(&self) -> IntDType {
        self.int_dtype
    }

    /// Sets the default floating-point data type.
    pub fn set_float_dtype(&mut self, dtype: impl Into<FloatDType>) {
        self.float_dtype = dtype.into();
    }

    /// Sets the default integer data type.
    pub fn set_int_dtype(&mut self, dtype: impl Into<IntDType>) {
        self.int_dtype = dtype.into();
    }

    /// Sets the default floating-point data type.
    pub fn with_float_dtype(mut self, dtype: impl Into<FloatDType>) -> Self {
        self.set_float_dtype(dtype);
        self
    }

    /// Sets the default integer data type.
    pub fn with_int_dtype(mut self, dtype: impl Into<IntDType>) -> Self {
        self.set_int_dtype(dtype);
        self
    }
}

impl Default for DevicePolicy {
    fn default() -> Self {
        Self {
            float_dtype: FloatDType::F32,
            int_dtype: IntDType::I32,
        }
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
/// Updates to the policy (via [`set_device_policy`], [`set_default_float_dtype`], etc.)
/// only affect future policy retrievals, not existing references.
///
/// This is intended for the common case where policies are set once during
/// initialization and then read frequently during tensor creation.
struct DevicePolicyRegistry;

impl DevicePolicyRegistry {
    /// Get the policy for a physical device type and device id.
    ///
    /// If no policy exists yet, a default one is created and stored.
    fn get_or_default<D: DeviceOps>(
        device: &D,
        // TODO: remove default fn once we move away from default backend elem types, and devices
        // might implement their own default policy
        default_fn: impl FnOnce() -> DevicePolicy,
    ) -> Arc<DevicePolicy> {
        let key = Self::key(device);

        if let Some(policy) = REGISTRY.read().unwrap().get(&key) {
            return Arc::clone(policy);
        }

        let mut map = REGISTRY.write().unwrap();
        Arc::clone(map.entry(key).or_insert_with(|| Arc::new(default_fn())))
    }

    /// Mutate the policy for a given device.
    fn update<D: DeviceOps>(
        device: &D,
        update_fn: impl FnOnce(&mut DevicePolicy),
        // TODO: remove default fn once we move away from default backend elem types, and devices
        // might implement their own default policy
        default_fn: impl FnOnce() -> DevicePolicy,
    ) {
        let key = Self::key(device);
        let mut map = REGISTRY.write().unwrap();

        let policy = map.entry(key).or_insert_with(|| Arc::new(default_fn()));

        // Update the policy
        let policy_mut = Arc::make_mut(policy);
        update_fn(policy_mut);
    }

    /// Returns the device registry key.
    fn key<D: Device>(device: &D) -> RegistryKey {
        (device.to_id(), TypeId::of::<D>())
    }
}

/// Default backend device policy.
/// Currently set to the backend's default float and int elem types for backward compat.
fn default_policy<B: Backend>() -> DevicePolicy {
    DevicePolicy::default()
        .with_float_dtype(B::FloatElem::dtype())
        .with_int_dtype(B::IntElem::dtype())
}

/// Get the [`device`'s policy](DevicePolicy).
///
/// Returns an immutable snapshot of the device's current policy. If the policy
/// is updated after retrieval, this snapshot will not reflect those changes.
pub(crate) fn get_device_policy<B: Backend>(device: &B::Device) -> Arc<DevicePolicy> {
    DevicePolicyRegistry::get_or_default(device, default_policy::<B>)
}

/// Sets the [`device`'s policy](DevicePolicy).
///
/// This updates the device's default data types used for tensor creation.
/// The policy should typically be set once during initialization and then
/// remains global for all subsequent operations on that device.
///
/// # Example
///
/// ```rust
/// use burn_tensor::backend::Backend;
/// use burn_tensor::{DevicePolicy, DType, Int, Tensor, set_device_policy};
///
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     
///     // Update the device policy
///     let policy = DevicePolicy::default()
///         .with_float_dtype(DType::F16)
///         .with_int_dtype(DType::I32);
///     set_device_policy::<B>(&device, policy);
///     
///     // All float tensors created after this will use F16 by default
///     let tensor = Tensor::<B, 2>::zeros([2, 3], &device);
///     // All int tensors created after this will use I32 default
///     let tensor = Tensor::<B, 2, Int>::zeros([2, 3], &device);
/// }
/// ```
pub fn set_device_policy<B: Backend>(device: &B::Device, policy: DevicePolicy) {
    DevicePolicyRegistry::update(
        device,
        |p| {
            p.set_float_dtype(policy.float_dtype());
            p.set_int_dtype(policy.int_dtype());
        },
        default_policy::<B>,
    );
}

/// Sets the default floating-point data type for the [device](DevicePolicy).
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
pub fn set_default_float_dtype<B: Backend>(device: &B::Device, dtype: impl Into<FloatDType>) {
    let dtype = dtype.into();

    DevicePolicyRegistry::update(
        device,
        |p| {
            p.set_float_dtype(dtype);
        },
        default_policy::<B>,
    );
}

/// Sets the default integer data type for the [device](DevicePolicy).
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
pub fn set_default_int_dtype<B: Backend>(device: &B::Device, dtype: impl Into<IntDType>) {
    let dtype = dtype.into();

    DevicePolicyRegistry::update(
        device,
        |p| {
            p.set_int_dtype(dtype);
        },
        default_policy::<B>,
    );
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

    fn get_test_device_policy<D: DeviceOps>(device: &D) -> Arc<DevicePolicy> {
        DevicePolicyRegistry::get_or_default(device, DevicePolicy::default)
    }

    fn set_test_device_policy<D: DeviceOps>(device: &D, policy: DevicePolicy) {
        DevicePolicyRegistry::update(
            device,
            |p| {
                p.set_float_dtype(policy.float_dtype());
                p.set_int_dtype(policy.int_dtype());
            },
            DevicePolicy::default,
        );
    }

    #[test]
    #[serial]
    fn default_policy_is_created_and_shared() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        let p1 = get_test_device_policy(&device);
        let p2 = get_test_device_policy(&device);

        assert!(Arc::ptr_eq(&p1, &p2));
        assert_eq!(p1.float_dtype(), FloatDType::F32);
        assert_eq!(p1.int_dtype(), IntDType::I32);
        assert_eq!(p2.float_dtype(), FloatDType::F32);
        assert_eq!(p2.int_dtype(), IntDType::I32);
    }

    #[test]
    #[serial]
    fn updated_policy_is_shared() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        // The device policy is meant to be set once at initialization
        set_test_device_policy(
            &device,
            DevicePolicy::default()
                .with_float_dtype(FloatDType::BF16)
                .with_int_dtype(IntDType::I32),
        );
        let p1 = get_test_device_policy(&device);
        let p2 = get_test_device_policy(&device);

        assert!(Arc::ptr_eq(&p1, &p2));
        assert_eq!(p1.float_dtype(), FloatDType::BF16);
        assert_eq!(p1.int_dtype(), IntDType::I32);
        assert_eq!(p2.float_dtype(), FloatDType::BF16);
        assert_eq!(p2.int_dtype(), IntDType::I32);
    }

    #[test]
    #[serial]
    fn policy_is_device_id_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceA::new(1);

        // set_default_float_dtype
        set_test_device_policy(
            &d1,
            DevicePolicy::default().with_float_dtype(FloatDType::F16),
        );

        let p1 = get_test_device_policy(&d1);
        let p2 = get_test_device_policy(&d2);

        assert!(!Arc::ptr_eq(&p1, &p2));
        assert_eq!(p1.float_dtype(), FloatDType::F16);
        assert_eq!(p1.int_dtype(), IntDType::I32);
        assert_eq!(p2.float_dtype(), FloatDType::F32);
        assert_eq!(p2.int_dtype(), IntDType::I32);
    }

    #[test]
    #[serial]
    fn policy_is_device_type_specific() {
        clear_registry(); // reset registry for each test

        let d1 = TestDeviceA::new(0);
        let d2 = TestDeviceB::new(0);

        // set_default_float_dtype
        set_test_device_policy(
            &d2,
            DevicePolicy::default().with_float_dtype(FloatDType::F16),
        );

        let p1 = get_test_device_policy(&d1);
        let p2 = get_test_device_policy(&d2);

        assert_eq!(p1.float_dtype(), FloatDType::F32);
        assert_eq!(p1.int_dtype(), IntDType::I32);
        assert_eq!(p2.float_dtype(), FloatDType::F16);
        assert_eq!(p2.int_dtype(), IntDType::I32);
    }

    #[test]
    #[serial]
    fn updating_policy_should_not_affect_snapshot() {
        clear_registry(); // reset registry for each test

        // The device policy is meant to be set once at initialization
        let device = TestDeviceA::new(0);
        let before = get_test_device_policy(&device);

        // set_default_float_dtype
        set_test_device_policy(
            &device,
            DevicePolicy::default().with_float_dtype(FloatDType::BF16),
        );

        let after = get_test_device_policy(&device);

        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(after.float_dtype(), FloatDType::BF16);
        assert_eq!(before.float_dtype(), FloatDType::F32);
    }

    #[test]
    #[serial]
    fn set_device_policy_overwrites_fields() {
        clear_registry(); // reset registry for each test

        let device = TestDeviceA::new(0);

        set_test_device_policy(
            &device,
            DevicePolicy::default()
                .with_float_dtype(FloatDType::F16)
                .with_int_dtype(IntDType::I64),
        );

        let policy = get_test_device_policy(&device);

        assert_eq!(policy.float_dtype(), FloatDType::F16);
        assert_eq!(policy.int_dtype(), IntDType::I64);
    }
}

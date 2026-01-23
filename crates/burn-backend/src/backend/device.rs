pub use burn_std::device::*;
use burn_std::{FloatDType, IntDType};

/// Device trait for all burn backend devices.
pub trait DeviceOps: Clone + Default + PartialEq + Send + Sync + core::fmt::Debug + Device {
    /// Returns the [device id](DeviceId).
    fn id(&self) -> DeviceId {
        self.to_id()
    }

    /// Returns the default policy used for tensor creation on this backend device.
    fn default_policy(&self) -> DevicePolicy {
        Default::default()
    }
}

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

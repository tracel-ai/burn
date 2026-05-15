//! Device data types: settings and errors.

use alloc::format;
use alloc::string::String;
use cubecl_common::backtrace::BackTrace;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{BoolDType, DType, FloatDType, IntDType};

/// Settings controlling the default data types for a specific device.
///
/// These settings are managed in a global registry that enforces strict initialization semantics:
///
/// 1. Manual Initialization: You can set these once at the start of your program using `set_default_dtypes`.
/// 2. Default Initialization: If an operation (like creating a tensor) occurs before manual initialization,
///    the settings are permanently locked to their default values.
/// 3. Immutability: Once initialized, settings cannot be changed. This ensures consistent behavior across
///    all threads and operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSettings {
    /// Default floating-point data type.
    pub float_dtype: FloatDType,
    /// Default integer data type.
    pub int_dtype: IntDType,
    /// Default bool data type.
    pub bool_dtype: BoolDType,
}

impl DeviceSettings {
    /// Creates a new [`DeviceSettings`] from any types convertible into the dtype kinds.
    pub fn new(
        float_dtype: impl Into<FloatDType>,
        int_dtype: impl Into<IntDType>,
        bool_dtype: impl Into<BoolDType>,
    ) -> Self {
        Self {
            float_dtype: float_dtype.into(),
            int_dtype: int_dtype.into(),
            bool_dtype: bool_dtype.into(),
        }
    }
}

/// Errors returned by device-related operations.
///
/// Examples include attempting to use an unsupported data type on a device or initialization
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
    pub fn unsupported_dtype<D: core::fmt::Debug + ?Sized>(device: &D, dtype: DType) -> Self {
        Self::UnsupportedDType {
            device: format!("{device:?}"),
            dtype,
        }
    }

    /// Helper to create a [`DeviceError::AlreadyInitialized`] from any device.
    pub fn already_initialized<D: core::fmt::Debug + ?Sized>(device: &D) -> Self {
        Self::AlreadyInitialized {
            device: format!("{device:?}"),
        }
    }
}

/// An error that can happen when syncing a device.
#[derive(Error, Serialize, Deserialize)]
pub enum ExecutionError {
    /// A generic error happened during execution.
    ///
    /// The backtrace and context information should be included in the reason string.
    #[error("An error happened during execution\nCaused by:\n  {reason}")]
    WithContext {
        /// The reason of the error.
        reason: String,
    },
    /// A generic error happened during execution thrown in the Burn project.
    ///
    /// The full context isn't captured by the string alone.
    #[error("An error happened during execution\nCaused by:\n  {reason}")]
    Generic {
        /// The reason of the error.
        reason: String,
        /// The backtrace.
        #[serde(skip)]
        backtrace: BackTrace,
    },
}

impl core::fmt::Debug for ExecutionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

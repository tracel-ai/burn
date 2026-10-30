//! Where a device's compiled kernels and autotune answers are kept.
//!
//! A compute environment holds what a device's work compiles and tunes, so the
//! next run reuses it rather than paying for it again. It lives in the storage
//! of the device that created it — a process's cache directory, a remote
//! server's — and is active on any device that reaches that storage. Entries
//! are kept apart by the hardware they were made for, so devices of one
//! architecture reuse each other's compiled kernels, devices of one model
//! each other's autotune answers, and unlike devices share an environment
//! without reading each other's.

use alloc::string::{String, ToString};
use burn_std::ExecutionError;
use burn_std::environment::EnvironmentError as StorageError;

pub use burn_std::environment::{EnvironmentBundle, EnvironmentId, EnvironmentOrigin};

/// Why a compute environment could not be created, activated or saved.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EnvironmentError {
    /// The backend keeps no compute environment: it compiles nothing, or
    /// keeps what it compiles nowhere.
    Unsupported {
        /// Why, in the backend's words.
        reason: String,
    },
    /// The device does not reach the storage the environment lives in:
    /// another server, or a device local to one that is not. Carry the
    /// environment over by saving it and creating one from the bundle.
    IncompatibleDevice {
        /// Why, in the backend's words.
        reason: String,
    },
    /// The environment cannot be replaced while a device's work uses it.
    Active {
        /// The environment's name.
        name: String,
    },
    /// The environment would be created from itself.
    SelfFork {
        /// The environment's name.
        name: String,
    },
    /// The storage refused a read or a write.
    Storage {
        /// Why, in the storage's words.
        reason: String,
    },
    /// The device could not be reached, or the work it held failed.
    Execution(ExecutionError),
}

impl From<StorageError> for EnvironmentError {
    fn from(error: StorageError) -> Self {
        match error {
            StorageError::IncompatibleStorage { .. } => Self::IncompatibleDevice {
                reason: error.to_string(),
            },
            StorageError::Active { name } => Self::Active { name },
            StorageError::SelfFork { name } => Self::SelfFork { name },
            StorageError::Storage { reason } => Self::Storage { reason },
        }
    }
}

impl From<ExecutionError> for EnvironmentError {
    fn from(error: ExecutionError) -> Self {
        Self::Execution(error)
    }
}

impl core::fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported { reason } => {
                write!(f, "the backend keeps no compute environment: {reason}")
            }
            Self::IncompatibleDevice { reason } => {
                write!(
                    f,
                    "the device cannot use this compute environment: {reason}"
                )
            }
            Self::Active { name } => write!(
                f,
                "the compute environment {name} is active, and cannot be replaced"
            ),
            Self::SelfFork { name } => write!(
                f,
                "the compute environment {name} cannot be created from itself"
            ),
            Self::Storage { reason } => write!(f, "the environment storage failed: {reason}"),
            Self::Execution(error) => write!(f, "{error}"),
        }
    }
}

impl core::error::Error for EnvironmentError {}

impl EnvironmentError {
    /// The answer of a backend that keeps no compute environment.
    pub(crate) fn unsupported() -> Self {
        Self::Unsupported {
            reason: String::from("it compiles nothing"),
        }
    }
}

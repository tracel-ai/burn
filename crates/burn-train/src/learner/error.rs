use burn_core::data::dataset::DatasetError;

use crate::{MetricsError, checkpoint::CheckpointerError, train::MultiDeviceStepError};

/// An error that stopped training or evaluation.
#[derive(Debug)]
pub enum TrainingError {
    /// Error during metrics processing.
    Metrics(MetricsError),
    /// Error while loading data.
    Dataset(DatasetError),
    /// The training step panicked on a worker of a multi-device strategy.
    Worker {
        /// The worker's device index.
        device_id: usize,
        /// The panic message.
        message: String,
    },
    /// Error while saving, restoring or deleting a checkpoint.
    Checkpoint(CheckpointerError),
}

impl TrainingError {
    /// Whether the error comes from a poisoned device. In this case, every later operation
    /// will fail until the process is restarted.
    pub fn is_device_poisoned(&self) -> bool {
        match self {
            Self::Metrics(err) => err.is_device_poisoned(),
            _ => false,
        }
    }
}

impl core::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Metrics(err) => write!(f, "{err}"),
            Self::Dataset(err) => write!(f, "Dataset error: {err}"),
            Self::Worker { device_id, message } => {
                write!(f, "Training worker on device {device_id} failed: {message}")
            }
            Self::Checkpoint(err) => write!(f, "Checkpoint error: {err}"),
        }
    }
}

impl core::error::Error for TrainingError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Metrics(err) => Some(err),
            Self::Dataset(err) => Some(err),
            Self::Worker { .. } => None,
            Self::Checkpoint(err) => Some(err),
        }
    }
}

impl From<MetricsError> for TrainingError {
    fn from(err: MetricsError) -> Self {
        Self::Metrics(err)
    }
}

impl From<DatasetError> for TrainingError {
    fn from(err: DatasetError) -> Self {
        Self::Dataset(err)
    }
}

impl From<CheckpointerError> for TrainingError {
    fn from(err: CheckpointerError) -> Self {
        Self::Checkpoint(err)
    }
}

impl From<MultiDeviceStepError> for TrainingError {
    fn from(err: MultiDeviceStepError) -> Self {
        match err {
            MultiDeviceStepError::Dataset(err) => Self::Dataset(err),
            MultiDeviceStepError::Worker { device_id, message } => {
                Self::Worker { device_id, message }
            }
        }
    }
}

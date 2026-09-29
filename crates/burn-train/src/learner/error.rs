use burn_core::data::dataset::DatasetError;

use crate::{
    MetricsError,
    checkpoint::CheckpointerError,
    train::{MultiDeviceStepError, WorkerFailure, fmt_worker_failures},
};

/// An error that stopped training or evaluation.
#[derive(Debug)]
pub enum TrainingError {
    /// Error during metrics processing.
    Metrics(MetricsError),
    /// Error while loading data.
    Dataset(DatasetError),
    /// The training step panicked on one or more workers of a multi-device strategy.
    Workers(Vec<WorkerFailure>),
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
            Self::Workers(failures) => fmt_worker_failures(failures, f),
            Self::Checkpoint(err) => write!(f, "Checkpoint error: {err}"),
        }
    }
}

impl core::error::Error for TrainingError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Metrics(err) => Some(err),
            Self::Dataset(err) => Some(err),
            Self::Workers(_) => None,
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
            MultiDeviceStepError::Workers(failures) => Self::Workers(failures),
        }
    }
}

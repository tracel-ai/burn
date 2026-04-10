use burn_core::{
    record::{Record, RecorderError},
    tensor::Device,
};
use thiserror::Error;

/// The error type for checkpointer.
#[derive(Error, Debug)]
pub enum CheckpointerError {
    /// IO error.
    #[error("I/O Error: `{0}`")]
    IOError(std::io::Error),

    /// Recorder error.
    #[error("Recorder error: `{0}`")]
    RecorderError(RecorderError),

    /// Other errors.
    #[error("Unknown error: `{0}`")]
    Unknown(String),
}

/// The trait for checkpointer.
pub trait Checkpointer<R>: Send + Sync
where
    R: Record,
{
    /// Save the record.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch.
    /// * `record` - The record.
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError>;

    /// Delete the record at the given epoch if present.
    fn delete(&self, epoch: usize) -> Result<(), CheckpointerError>;

    /// Restore the record.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch.
    /// * `device` - The device used to restore the record.
    ///
    /// # Returns
    ///
    /// The record.
    fn restore(&self, epoch: usize, device: &Device) -> Result<R, CheckpointerError>;
}

use burn_core::{
    record::{Record, RecorderError},
    tensor::backend::Backend,
};

/// The error type for checkpointer.
#[derive(Debug)]
pub enum CheckpointerError {
    /// IO error.
    IOError(std::io::Error),

    /// Recorder error.
    RecorderError(RecorderError),

    /// Other errors.
    Unknown(String),
}

/// The trait for checkpointer.
pub trait Checkpointer<R, B>: Send + Sync
where
    R: Record<B>,
    B: Backend,
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
    fn restore(&self, epoch: usize, device: &B::Device) -> Result<R, CheckpointerError>;
}

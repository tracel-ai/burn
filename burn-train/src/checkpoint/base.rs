use burn_core::record::{Record, RecorderError};

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
pub trait Checkpointer<R: Record> {
    /// Save the record.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch.
    /// * `record` - The record.
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError>;

    /// Restore the record.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch.
    ///
    /// # Returns
    ///
    /// The record.
    fn restore(&self, epoch: usize) -> Result<R, CheckpointerError>;
}

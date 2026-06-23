use burn_core::store::{ModuleRecord, RecordError};
use burn_optim::OptimizerRecord;
use burn_optim::lr_scheduler::LrSchedulerRecord;
use burn_std::Bytes;
use std::path::PathBuf;
use thiserror::Error;

/// The error type for checkpointer.
#[derive(Error, Debug)]
pub enum CheckpointerError {
    /// IO error.
    #[error("I/O Error: `{0}`")]
    IOError(std::io::Error),

    /// Record (burnpack) error.
    #[error("Record error: `{0}`")]
    Record(RecordError),

    /// Other errors.
    #[error("Unknown error: `{0}`")]
    Unknown(String),
}

/// A record that can be saved to and loaded from a burnpack file.
///
/// Implemented for the burnpack record types used during training: the module
/// ([`ModuleRecord`]), the optimizer ([`OptimizerRecord`]) and the learning rate scheduler
/// ([`LrSchedulerRecord`]).
///
/// Records are device-free: a checkpoint is just file-backed bytes. Device placement is decided
/// when a record is applied (the module keeps its existing parameter device; optimizer state
/// migrates to each parameter's device on the next step), not when the checkpoint is loaded.
pub trait Checkpoint: Sized + Send + 'static {
    /// Save the record to `path`.
    fn save(self, path: PathBuf) -> Result<(), CheckpointerError>;
    /// Load the record from `path`.
    fn load(path: PathBuf) -> Result<Self, CheckpointerError>;
    /// Creates a checkpoint from bytes
    fn checkpoint_from_bytes(bytes: Bytes) -> Result<Self, RecordError>;
    /// Transforms a checkpoint into bytes
    fn checkpoint_into_bytes(self) -> Result<Bytes, RecordError>;

    /// Stream the checkpoint to any [`std::io::Write`] without materializing the whole buffer.
    ///
    /// The default materializes via [`checkpoint_into_bytes`](Self::checkpoint_into_bytes); the
    /// burnpack record types override it to stream tensor-by-tensor.
    fn checkpoint_into_writer<W: std::io::Write>(self, mut writer: W) -> Result<(), RecordError> {
        let bytes = self.checkpoint_into_bytes()?;
        std::io::Write::write_all(&mut writer, &bytes).map_err(|e| RecordError::Io(e.to_string()))
    }

    /// Reconstruct the checkpoint by streaming from any [`std::io::Read`].
    ///
    /// The default materializes via [`checkpoint_from_bytes`](Self::checkpoint_from_bytes); the
    /// burnpack record types override it to stream.
    fn checkpoint_from_reader<R: std::io::Read>(mut reader: R) -> Result<Self, RecordError> {
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut buf)
            .map_err(|e| RecordError::Io(e.to_string()))?;
        Self::checkpoint_from_bytes(Bytes::from_bytes_vec(buf))
    }
}

/// A stateless record: nothing to save or load.
impl Checkpoint for () {
    fn save(self, _path: PathBuf) -> Result<(), CheckpointerError> {
        Ok(())
    }
    fn load(_path: PathBuf) -> Result<Self, CheckpointerError> {
        Ok(())
    }
    fn checkpoint_from_bytes(_bytes: Bytes) -> Result<Self, RecordError> {
        Ok(())
    }
    fn checkpoint_into_bytes(self) -> Result<Bytes, RecordError> {
        Ok(Bytes::from_bytes_vec(vec![0]))
    }
}

impl Checkpoint for ModuleRecord {
    fn save(self, path: PathBuf) -> Result<(), CheckpointerError> {
        ModuleRecord::save(self, path).map_err(CheckpointerError::Record)
    }
    fn load(path: PathBuf) -> Result<Self, CheckpointerError> {
        ModuleRecord::load(path).map_err(CheckpointerError::Record)
    }
    fn checkpoint_into_bytes(self) -> Result<Bytes, RecordError> {
        self.into_bytes()
    }
    fn checkpoint_from_bytes(bytes: Bytes) -> Result<Self, RecordError> {
        ModuleRecord::from_bytes(bytes)
    }
    fn checkpoint_into_writer<W: std::io::Write>(self, writer: W) -> Result<(), RecordError> {
        self.into_writer(writer)
    }
    fn checkpoint_from_reader<R: std::io::Read>(reader: R) -> Result<Self, RecordError> {
        ModuleRecord::from_reader(reader)
    }
}

impl Checkpoint for OptimizerRecord {
    fn save(self, path: PathBuf) -> Result<(), CheckpointerError> {
        OptimizerRecord::save(self, path).map_err(CheckpointerError::Record)
    }
    fn load(path: PathBuf) -> Result<Self, CheckpointerError> {
        OptimizerRecord::load(path).map_err(CheckpointerError::Record)
    }
    fn checkpoint_from_bytes(bytes: Bytes) -> Result<Self, RecordError> {
        OptimizerRecord::from_bytes(bytes)
    }
    fn checkpoint_into_bytes(self) -> Result<Bytes, RecordError> {
        self.into_bytes()
    }
    fn checkpoint_into_writer<W: std::io::Write>(self, writer: W) -> Result<(), RecordError> {
        self.into_writer(writer)
    }
    fn checkpoint_from_reader<R: std::io::Read>(reader: R) -> Result<Self, RecordError> {
        OptimizerRecord::from_reader(reader)
    }
}

impl Checkpoint for LrSchedulerRecord {
    fn save(self, path: PathBuf) -> Result<(), CheckpointerError> {
        LrSchedulerRecord::save(self, path).map_err(CheckpointerError::Record)
    }
    fn load(path: PathBuf) -> Result<Self, CheckpointerError> {
        LrSchedulerRecord::load(path).map_err(CheckpointerError::Record)
    }
    fn checkpoint_from_bytes(bytes: Bytes) -> Result<Self, RecordError> {
        LrSchedulerRecord::from_bytes(bytes)
    }
    fn checkpoint_into_bytes(self) -> Result<Bytes, RecordError> {
        self.into_bytes()
    }
    fn checkpoint_into_writer<W: std::io::Write>(self, writer: W) -> Result<(), RecordError> {
        self.into_writer(writer)
    }
    fn checkpoint_from_reader<R: std::io::Read>(reader: R) -> Result<Self, RecordError> {
        LrSchedulerRecord::from_reader(reader)
    }
}

/// The trait for checkpointer.
pub trait Checkpointer<R>: Send + Sync
where
    R: Checkpoint,
{
    /// Save the record.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch.
    /// * `record` - The record.
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError>;

    /// Delete the checkpoint saved at the given epoch if present.
    fn delete(&self, epoch: usize) -> Result<(), CheckpointerError>;

    /// Restore the record from the checkpoint saved at the given epoch.
    fn restore(&self, epoch: usize) -> Result<R, CheckpointerError>;
}

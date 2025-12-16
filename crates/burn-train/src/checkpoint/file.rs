use std::path::{Path, PathBuf};

use super::{Checkpointer, CheckpointerError};
use burn_core::{
    record::{FileRecorder, Record},
    tensor::backend::Backend,
};

/// The file checkpointer.
pub struct FileCheckpointer<FR> {
    directory: PathBuf,
    name: String,
    recorder: FR,
}

impl<FR> FileCheckpointer<FR> {
    /// Creates a new file checkpointer.
    ///
    /// # Arguments
    ///
    /// * `recorder` - The file recorder.
    /// * `directory` - The directory to save the checkpoints.
    /// * `name` - The name of the checkpoint.
    pub fn new(recorder: FR, directory: impl AsRef<Path>, name: &str) -> Self {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory).ok();

        Self {
            directory: directory.to_path_buf(),
            name: name.to_string(),
            recorder,
        }
    }

    fn path_for_epoch(&self, epoch: usize) -> PathBuf {
        self.directory.join(format!("{}-{}", self.name, epoch))
    }
}

impl<FR, R, B> Checkpointer<R, B> for FileCheckpointer<FR>
where
    R: Record<B>,
    FR: FileRecorder<B>,
    B: Backend,
{
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);
        log::trace!("Saving checkpoint {} to {}", epoch, file_path.display());

        self.recorder
            .record(record, file_path)
            .map_err(CheckpointerError::RecorderError)?;

        Ok(())
    }

    fn restore(&self, epoch: usize, device: &B::Device) -> Result<R, CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);
        log::info!(
            "Restoring checkpoint {} from {}",
            epoch,
            file_path.display()
        );
        let record = self
            .recorder
            .load(file_path, device)
            .map_err(CheckpointerError::RecorderError)?;

        Ok(record)
    }

    fn delete(&self, epoch: usize) -> Result<(), CheckpointerError> {
        let file_to_remove = format!(
            "{}.{}",
            self.path_for_epoch(epoch).display(),
            FR::file_extension(),
        );

        if std::path::Path::new(&file_to_remove).exists() {
            log::trace!("Removing checkpoint {file_to_remove}");
            std::fs::remove_file(file_to_remove).map_err(CheckpointerError::IOError)?;
        }

        Ok(())
    }
}

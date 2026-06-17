use std::path::{Path, PathBuf};

use super::{Checkpoint, Checkpointer, CheckpointerError};

/// The file checkpointer.
///
/// Saves each record as a [burnpack](burn_core::store) file in the given directory.
pub struct FileCheckpointer {
    directory: PathBuf,
    name: String,
}

impl FileCheckpointer {
    /// Creates a new file checkpointer.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    /// * `name` - The name of the checkpoint.
    pub fn new(directory: impl AsRef<Path>, name: &str) -> Self {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory).ok();

        Self {
            directory: directory.to_path_buf(),
            name: name.to_string(),
        }
    }

    fn path_for_epoch(&self, epoch: usize) -> PathBuf {
        self.directory.join(format!("{}-{}.bpk", self.name, epoch))
    }
}

impl<R> Checkpointer<R> for FileCheckpointer
where
    R: Checkpoint,
{
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);
        log::trace!("Saving checkpoint {} to {}", epoch, file_path.display());

        record.save(file_path)
    }

    fn restore(&self, epoch: usize) -> Result<R, CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);
        log::info!(
            "Restoring checkpoint {} from {}",
            epoch,
            file_path.display()
        );

        R::load(file_path)
    }

    fn delete(&self, epoch: usize) -> Result<(), CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);

        if file_path.exists() {
            log::trace!("Removing checkpoint {}", file_path.display());
            std::fs::remove_file(file_path).map_err(CheckpointerError::IOError)?;
        }

        Ok(())
    }
}

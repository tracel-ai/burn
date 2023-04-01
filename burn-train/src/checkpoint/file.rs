use std::marker::PhantomData;

use super::{Checkpointer, CheckpointerError};
use burn_core::record::{FileRecorder, Record, RecordSettings};
use serde::{de::DeserializeOwned, Serialize};

pub struct FileCheckpointer<S>
where
    S: RecordSettings,
    S::Recorder: FileRecorder,
{
    directory: String,
    name: String,
    num_keep: usize,
    settings: PhantomData<S>,
}

impl<S> FileCheckpointer<S>
where
    S: RecordSettings,
    S::Recorder: FileRecorder,
{
    pub fn new(directory: &str, name: &str, num_keep: usize) -> Self {
        std::fs::create_dir_all(directory).ok();

        Self {
            directory: directory.to_string(),
            name: name.to_string(),
            num_keep,
            settings: PhantomData::default(),
        }
    }
    fn path_for_epoch(&self, epoch: usize) -> String {
        format!("{}/{}-{}", self.directory, self.name, epoch)
    }
}

impl<R, S> Checkpointer<R> for FileCheckpointer<S>
where
    R: Record,
    S: RecordSettings,
    S::Recorder: FileRecorder,
    R::Item<S>: Serialize + DeserializeOwned,
{
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);
        log::info!("Saving checkpoint {} to {}", epoch, file_path);

        record
            .record(file_path.into())
            .map_err(CheckpointerError::RecorderError)?;

        if self.num_keep > epoch {
            return Ok(());
        }

        let file_path_old_checkpoint = self.path_for_epoch(epoch - self.num_keep);

        if std::path::Path::new(&file_path_old_checkpoint).exists() {
            log::info!("Removing checkpoint {}", file_path_old_checkpoint);
            std::fs::remove_file(file_path_old_checkpoint).map_err(CheckpointerError::IOError)?;
        }

        Ok(())
    }

    fn restore(&self, epoch: usize) -> Result<R, CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);
        log::info!("Restoring checkpoint {} from {}", epoch, file_path);
        let record = R::load(file_path.into()).map_err(CheckpointerError::RecorderError)?;

        Ok(record)
    }
}

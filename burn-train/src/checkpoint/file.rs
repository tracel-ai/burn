use super::{Checkpointer, CheckpointerError};
use burn_core::module::State;
use burn_core::tensor::Element;

pub struct FileCheckpointer<P> {
    directory: String,
    name: String,
    num_keep: usize,
    _precision: P,
}

impl<P: Element> FileCheckpointer<P> {
    pub fn new(directory: &str, name: &str, num_keep: usize) -> Self {
        std::fs::create_dir_all(directory).ok();

        Self {
            directory: directory.to_string(),
            name: name.to_string(),
            num_keep,
            _precision: P::default(),
        }
    }
    fn path_for_epoch(&self, epoch: usize) -> String {
        format!("{}/{}-{}.json.gz", self.directory, self.name, epoch)
    }
}

impl<E, P> Checkpointer<E> for FileCheckpointer<P>
where
    P: serde::Serialize + serde::de::DeserializeOwned + Element,
    E: Element,
{
    fn save(&self, epoch: usize, state: State<E>) -> Result<(), CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);
        log::info!("Saving checkpoint {} to {}", epoch, file_path);

        state
            .convert::<P>()
            .save(&file_path)
            .map_err(CheckpointerError::IOError)?;

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

    fn restore(&self, epoch: usize) -> Result<State<E>, CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);
        log::info!("Restoring checkpoint {} from {}", epoch, file_path);

        let state = State::<P>::load(&file_path).map_err(CheckpointerError::StateError)?;

        Ok(state.convert())
    }
}

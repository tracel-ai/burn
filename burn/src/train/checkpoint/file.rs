use super::{Checkpointer, CheckpointerError};
use crate::module::State;
use burn_tensor::Element;

pub struct FileCheckpointer<P> {
    directory: String,
    name: String,
    _precision: P,
}

impl<P: Element> FileCheckpointer<P> {
    pub fn new(directory: &str, name: &str) -> Self {
        std::fs::create_dir(directory).ok();

        Self {
            directory: directory.to_string(),
            name: name.to_string(),
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
        state
            .convert::<P>()
            .save(&file_path)
            .map_err(CheckpointerError::IOError)?;

        let file_path_old_checkpoint = self.path_for_epoch(epoch - 1);

        if std::path::Path::new(&file_path_old_checkpoint).exists() {
            std::fs::remove_file(file_path_old_checkpoint).map_err(CheckpointerError::IOError)?;
        }

        Ok(())
    }

    fn restore(&self, epoch: usize) -> Result<State<E>, CheckpointerError> {
        let file_path = self.path_for_epoch(epoch);

        let state = State::<P>::load(&file_path).map_err(CheckpointerError::StateError)?;

        Ok(state.convert())
    }
}

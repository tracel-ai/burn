use core::marker::PhantomData;
use std::path::PathBuf;

use burn::record::{BurnRecord, FileRecorder, PrecisionSettings, Record, Recorder, RecorderError};
use serde::{de::DeserializeOwned, Serialize};

use super::de::from_file;

#[derive(new, Debug, Default, Clone)]
pub struct PyTorchFileRecorder<S: PrecisionSettings> {
    _settings: PhantomData<S>,
}

impl<S: PrecisionSettings> Recorder for PyTorchFileRecorder<S> {
    type Settings = S;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn save_item<I: Serialize>(
        &self,
        _item: I,
        _file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        unimplemented!("save_item not implemented for PyTorchFileRecorder")
    }

    fn load_item<I: DeserializeOwned>(&self, file: Self::LoadArgs) -> Result<I, RecorderError> {
        // let state = from_file::<I>(&file)?;
        // Ok(state)
        unimplemented!("load_item not implemented for PyTorchFileRecorder")
    }

    fn load<R: Record>(&self, args: Self::LoadArgs) -> Result<R, RecorderError> {
        let item = from_file::<R::Item<Self::Settings>>(&args)?;
        Ok(R::from_item(item))
    }
}

impl<S: PrecisionSettings> FileRecorder for PyTorchFileRecorder<S> {
    fn file_extension() -> &'static str {
        "pt"
    }
}

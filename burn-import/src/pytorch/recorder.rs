use core::marker::PhantomData;
use std::path::PathBuf;

use burn::record::{PrecisionSettings, Record, Recorder, RecorderError};
use regex::Regex;
use serde::{de::DeserializeOwned, Serialize};

use super::de::from_file;

/// A recorder that that loads PyTorch files (`.pt`).
#[derive(new, Debug, Default, Clone)]
pub struct PyTorchFileRecorder<PS: PrecisionSettings> {
    _settings: PhantomData<PS>,
}

impl<PS: PrecisionSettings> Recorder for PyTorchFileRecorder<PS> {
    type Settings = PS;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = LoadArgs;

    fn save_item<I: Serialize>(
        &self,
        _item: I,
        _file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        unimplemented!("save_item not implemented for PyTorchFileRecorder")
    }

    fn load_item<I: DeserializeOwned>(&self, _file: Self::LoadArgs) -> Result<I, RecorderError> {
        unimplemented!("load_item not implemented for PyTorchFileRecorder")
    }

    fn load<R: Record>(&self, args: Self::LoadArgs) -> Result<R, RecorderError> {
        let item = from_file::<PS, R::Item<Self::Settings>>(&args.file).unwrap();
        Ok(R::from_item(item))
    }
}

#[derive(Debug, Clone)]
pub struct LoadArgs {
    pub file: PathBuf,
    pub key_remap: Vec<(Regex, String)>,
}

impl LoadArgs {
    pub fn new(file: PathBuf) -> Self {
        Self {
            file,
            key_remap: Vec::new(),
        }
    }

    pub fn with_key_remap(mut self, key_remap: Vec<(Regex, String)>) -> Self {
        self.key_remap = key_remap;
        self
    }
}

impl From<PathBuf> for LoadArgs {
    fn from(val: PathBuf) -> Self {
        LoadArgs::new(val)
    }
}

impl From<String> for LoadArgs {
    fn from(val: String) -> Self {
        LoadArgs::new(val.into())
    }
}

impl From<&str> for LoadArgs {
    fn from(val: &str) -> Self {
        LoadArgs::new(val.into())
    }
}

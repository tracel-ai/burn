use core::marker::PhantomData;
use std::path::PathBuf;

use burn::record::{PrecisionSettings, Record, Recorder, RecorderError};

use regex::Regex;
use serde::{de::DeserializeOwned, Serialize};

use super::reader::from_file;

/// A recorder that that loads PyTorch files (`.pt`) into Burn modules.
///
/// LoadArgs can be used to remap keys or file path.
/// See [LoadArgs](struct.LoadArgs.html) for more information.
///
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
        let item = from_file::<PS, R::Item<Self::Settings>>(&args.file, args.key_remap).unwrap();
        Ok(R::from_item(item))
    }
}

/// Arguments for loading a PyTorch file.
///
/// # Examples
///
/// ```no_run
/// use burn_import::pytorch::LoadArgs;
///
/// let args = LoadArgs::new("tests/key_remap/key_remap.pt".into())
///    .with_key_remap("conv\\.(.*)", "$1"); // // Remove "conv" prefix, e.g. "conv.conv1" -> "conv1"
///
/// let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
///   .load(args)
///  .expect("Failed to decode state");
/// ```
#[derive(Debug, Clone)]
pub struct LoadArgs {
    /// The path to the file to load.
    pub file: PathBuf,

    /// A list of key remappings.
    pub key_remap: Vec<(Regex, String)>,
}

impl LoadArgs {
    /// Create a new `LoadArgs` instance.
    ///
    /// # Arguments
    ///
    /// * `file` - The path to the file to load.
    pub fn new(file: PathBuf) -> Self {
        Self {
            file,
            key_remap: Vec::new(),
        }
    }

    /// Set key remapping.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The Regex pattern to be replaced.
    /// * `replacement` - The pattern to replace with.
    ///
    /// See [Regex](https://docs.rs/regex/1.5.4/regex/#syntax) for the pattern syntax and
    /// [Replacement](https://docs.rs/regex/latest/regex/struct.Regex.html#method.replace) for the
    /// replacement syntax.
    pub fn with_key_remap(mut self, pattern: &str, replacement: &str) -> Self {
        let regex = Regex::new(&format!("^{}$", pattern)).unwrap();

        self.key_remap.push((regex, replacement.into()));
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

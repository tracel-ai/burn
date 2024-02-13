use core::marker::PhantomData;
use std::path::PathBuf;

use burn::{
    record::{PrecisionSettings, Record, Recorder, RecorderError},
    tensor::backend::Backend,
};

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

impl<PS: PrecisionSettings, B: Backend> Recorder<B> for PyTorchFileRecorder<PS> {
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

    fn load<R: Record<B>>(
        &self,
        args: Self::LoadArgs,
        device: &B::Device,
    ) -> Result<R, RecorderError> {
        let item = from_file::<PS, R::Item<Self::Settings>, B>(
            &args.file,
            args.key_remap,
            args.top_level_key.as_deref(), // Convert Option<String> to Option<&str>
        )?;
        Ok(R::from_item(item, device))
    }
}

/// Arguments for loading a PyTorch file.
///
/// # Fields
///
/// * `file` - The path to the file to load.
/// * `key_remap` - A vector of tuples containing a regular expression and a replacement string.
///                See [regex::Regex::replace](https://docs.rs/regex/latest/regex/struct.Regex.html#method.replace)
///                for more information.
///
/// # Notes
///
/// Use [Netron](https://github.com/lutzroeder/netron) to inspect the keys of the PyTorch file (.pt extension).
///
///
/// # Examples
///
/// ```text
/// use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
/// use burn::record::FullPrecisionSettings;
/// use burn::record::Recorder;
///
/// let args = LoadArgs::new("tests/key_remap/key_remap.pt".into())
///    .with_key_remap("conv\\.(.*)", "$1"); // // Remove "conv" prefix, e.g. "conv.conv1" -> "conv1"
///
/// let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
///   .load(args)
///   .expect("Should decode state successfully");
/// ```
#[derive(Debug, Clone)]
pub struct LoadArgs {
    /// The path to the file to load.
    pub file: PathBuf,

    /// A list of key remappings.
    pub key_remap: Vec<(Regex, String)>,

    /// Top-level key to load state_dict from the file.
    /// Sometimes the state_dict is nested under a top-level key in a dict.
    pub top_level_key: Option<String>,
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
            top_level_key: None,
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
        let regex = Regex::new(pattern).expect("Valid regex");

        self.key_remap.push((regex, replacement.into()));
        self
    }

    /// Set top-level key to load state_dict from the file.
    /// Sometimes the state_dict is nested under a top-level key in a dict.
    ///
    /// # Arguments
    ///
    /// * `key` - The top-level key to load state_dict from the file.
    pub fn with_top_level_key(mut self, key: &str) -> Self {
        self.top_level_key = Some(key.into());
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

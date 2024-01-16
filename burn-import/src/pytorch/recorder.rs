use core::marker::PhantomData;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use burn::record::{PrecisionSettings, Record, Recorder, RecorderError};
use candle_core::{Tensor as CandleTensor, pickle};

use regex::Regex;
use serde::{de::DeserializeOwned, Serialize};

use crate::record::{de::Deserializer, error::Error};

use super::{target_file::reverse_flatten, adapter::PyTorchAdapter};

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

pub fn from_file<PS, D>(path: &Path, key_remap: Vec<(Regex, String)>) -> Result<D, Error>
where
    D: DeserializeOwned,
    PS: PrecisionSettings,
{
    // Read the pickle file and return a vector of Candle tensors
    let tensors: HashMap<String, CandleTensor> =
        pickle::read_all(path).unwrap().into_iter().collect();

    // Remap the keys (replace the keys in the map with the new keys)
    let tensors = remap(tensors, key_remap);

    // Convert the vector of Candle tensors to a nested map/vector of tensors
    let nested_value = reverse_flatten::<PS>(tensors);

    let deserializer = Deserializer::<PyTorchAdapter<PS>>::new(nested_value);

    let value = D::deserialize(deserializer)?;
    Ok(value)
}

/// Remap the tensor locations according to the key remapping.
fn remap(
    mut tensors: HashMap<String, CandleTensor>,
    key_remap: Vec<(Regex, String)>,
) -> HashMap<String, CandleTensor> {
    if key_remap.is_empty() {
        return tensors;
    }

    let mut remapped = HashMap::new();

    for (name, tensor) in tensors.drain() {
        let mut new_name = name.clone();
        for (pattern, replacement) in &key_remap {
            if pattern.is_match(&name) {
                new_name = pattern.replace_all(&name, replacement.as_str()).to_string();
                break;
            }
        }
        remapped.insert(new_name, tensor);
    }

    remapped
}

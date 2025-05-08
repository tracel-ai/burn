use core::marker::PhantomData;
use std::path::PathBuf;

use burn::{
    record::{PrecisionSettings, Record, Recorder, RecorderError},
    tensor::backend::Backend,
};

use regex::Regex;
use serde::{Serialize, de::DeserializeOwned};

use super::reader::from_file;

/// Recorder for loading PyTorch (`.pt`) files into Burn modules.
///
/// Load arguments ([`LoadArgs`]) can be used to specify the file path and
/// remap parameter keys during loading.
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
        unimplemented!("Save operations are not supported by PyTorchFileRecorder.")
    }

    fn load_item<I: DeserializeOwned>(
        &self,
        _file: &mut Self::LoadArgs,
    ) -> Result<I, RecorderError> {
        unimplemented!("load_item is not implemented for PyTorchFileRecorder; use load instead.")
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
            args.debug,
        )?;
        Ok(R::from_item(item, device))
    }
}

/// Arguments for loading PyTorch model weights.
///
/// # Notes
///
/// Parameter keys within a PyTorch file (`.pt` extension) can be inspected using
/// tools like [Netron](https://github.com/lutzroeder/netron).
///
/// # Examples
///
/// ```rust,ignore
/// use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
/// use burn::record::{FullPrecisionSettings, Recorder};
///
/// // Create load arguments, specifying the file and a key remapping rule.
/// let args = LoadArgs::new("tests/key_remap/key_remap.pt".into())
///    // Remove "conv." prefix, e.g., "conv.weight" -> "weight"
///    .with_key_remap("conv\\.(.*)", "$1");
///
/// // Load the record using the default recorder.
/// let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
///   .load(args, &burn::backend::NdArray::default().device()) // Provide a device
///   .expect("Failed to decode state from file"); // Example assertion
/// ```
#[derive(Debug, Clone)]
pub struct LoadArgs {
    /// The path to the PyTorch file (`.pt`).
    pub file: PathBuf,

    /// A list of key remapping rules applied to the state dictionary keys.
    /// Each rule consists of a regular expression and a replacement string.
    /// See [regex::Regex::replace](https://docs.rs/regex/latest/regex/struct.Regex.html#method.replace)
    /// for more details.
    pub key_remap: Vec<(Regex, String)>,

    /// Optional top-level key under which the state dictionary is nested within the file.
    /// If `None`, the root object is assumed to be the state dictionary.
    pub top_level_key: Option<String>,

    /// If `true`, prints debug information during the loading process.
    pub debug: bool,
}

impl LoadArgs {
    /// Creates new load arguments with the given file path.
    ///
    /// # Arguments
    ///
    /// * `file` - The path to the PyTorch file to load.
    pub fn new(file: PathBuf) -> Self {
        Self {
            file,
            key_remap: Vec::new(),
            top_level_key: None,
            debug: false,
        }
    }

    /// Adds a key remapping rule.
    ///
    /// Keys from the PyTorch state dictionary are modified if they match the pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The regular expression pattern to match against state dictionary keys.
    /// * `replacement` - The replacement string. Capture groups can be used (e.g., `$1`).
    ///
    /// # Panics
    ///
    /// Panics if the provided `pattern` is an invalid regular expression.
    ///
    /// See the [regex crate documentation](https://docs.rs/regex/latest/regex/) for pattern syntax
    /// and [replacement string syntax](https://docs.rs/regex/latest/regex/struct.Regex.html#replacement-string-syntax).
    pub fn with_key_remap(mut self, pattern: &str, replacement: &str) -> Self {
        let regex = Regex::new(pattern).expect("Invalid regex pattern provided to with_key_remap");
        self.key_remap.push((regex, replacement.into()));
        self
    }

    /// Specifies a top-level key in the file under which the state dictionary is nested.
    ///
    /// Some PyTorch files store the state dictionary within a larger structure (e.g., a dictionary).
    /// Use this method if the weights are not at the root level of the file.
    ///
    /// # Arguments
    ///
    /// * `key` - The top-level key to access the state dictionary.
    pub fn with_top_level_key(mut self, key: &str) -> Self {
        self.top_level_key = Some(key.into());
        self
    }

    /// Enables printing of debug information during loading.
    pub fn with_debug_print(mut self) -> Self {
        self.debug = true;
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

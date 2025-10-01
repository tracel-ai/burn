use core::marker::PhantomData;
use std::path::PathBuf;

use burn::{
    record::{PrecisionSettings, Record, Recorder, RecorderError},
    tensor::backend::Backend,
};

use regex::Regex;
use serde::{Serialize, de::DeserializeOwned};

use super::reader::from_file;

/// Recorder for loading HuggingFace Safetensors files (`.safetensors`) into Burn modules.
///
/// This recorder uses [LoadArgs] to configure loading behavior, such as key remapping.
#[derive(new, Debug, Default, Clone)]
pub struct SafetensorsFileRecorder<PS: PrecisionSettings> {
    _settings: PhantomData<PS>,
}

impl<PS: PrecisionSettings, B: Backend> Recorder<B> for SafetensorsFileRecorder<PS> {
    type Settings = PS;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = LoadArgs;

    fn save_item<I: Serialize>(
        &self,
        _item: I,
        _file: Self::RecordArgs,
    ) -> Result<(), RecorderError> {
        unimplemented!("save_item not implemented for SafetensorsFileRecorder")
    }

    fn load_item<I: DeserializeOwned>(
        &self,
        _file: &mut Self::LoadArgs,
    ) -> Result<I, RecorderError> {
        unimplemented!("load_item not implemented for SafetensorsFileRecorder")
    }

    fn load<R: Record<B>>(
        &self,
        args: Self::LoadArgs,
        device: &B::Device,
    ) -> Result<R, RecorderError> {
        let item = from_file::<PS, R::Item<Self::Settings>, B>(
            &args.file,
            args.key_remap,
            args.debug,
            args.adapter_type,
        )?;
        Ok(R::from_item(item, device))
    }
}

/// Arguments for loading a Safetensors file using [SafetensorsFileRecorder].
///
/// # Example
///
/// ```rust,ignore
/// use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};
/// use burn::record::{FullPrecisionSettings, Recorder};
/// use std::path::PathBuf;
///
/// // Dummy model record structure
/// #[derive(Record, Default)]
/// struct MyModelRecord<B: Backend> {
///     // fields matching the tensor names in the file
/// }
///
/// let device = Default::default(); // Replace with your actual device
///
/// // Example assuming a file named 'model.safetensors' exists
/// let args = LoadArgs::new(PathBuf::from("model.safetensors"))
///    // Example: Remove "model.encoder." prefix from keys
///    .with_key_remap("model\\.encoder\\.(.*)", "$1")
///    .with_adapter_type(AdapterType::PyTorch) // Specify if adaptation is needed
///    .with_debug_print(); // Enable debug output
///
/// let record: MyModelRecord<MyBackend> = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
///    .load(args, &device)
///    .expect("Should decode state successfully");
/// ```
#[derive(Debug, Clone)]
pub struct LoadArgs {
    /// The path to the Safetensors file to load.
    pub file: PathBuf,

    /// A list of key remapping rules applied sequentially. Each tuple contains a
    /// regular expression ([`Regex`]) to match keys and a replacement string.
    /// See [regex::Regex::replace_all](https://docs.rs/regex/latest/regex/struct.Regex.html#method.replace_all)
    /// for replacement syntax details.
    pub key_remap: Vec<(Regex, String)>,

    /// If true, prints debug information during the loading process.
    pub debug: bool,

    /// The type of adapter to apply for potential framework-specific tensor transformations
    /// (e.g., transposing certain weights).
    pub adapter_type: AdapterType,
}

/// Specifies the type of adapter to use for tensor loading.
///
/// Adapters handle potential differences in tensor formats or naming conventions
/// between the source framework and Burn.
#[derive(Debug, Clone, Default, Copy)]
pub enum AdapterType {
    /// Adapts tensors assuming they originated from PyTorch.
    #[default]
    PyTorch,

    /// Loads tensors directly without any specific adaptation.
    NoAdapter,
}

impl LoadArgs {
    /// Creates new `LoadArgs` for the given file path.
    ///
    /// By default, no key remapping is applied, debug printing is off,
    /// and the adapter type is [AdapterType::PyTorch].
    ///
    /// # Arguments
    ///
    /// * `file` - The path to the Safetensors file.
    pub fn new(file: PathBuf) -> Self {
        Self {
            file,
            key_remap: Vec::new(),
            debug: false,
            adapter_type: Default::default(),
        }
    }

    /// Adds a key remapping rule.
    ///
    /// Rules are applied in the order they are added.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The regular expression pattern to match tensor keys.
    /// * `replacement` - The replacement string. Capture groups like `$1`, `$2` can be used.
    ///
    /// # Panics
    ///
    /// Panics if `pattern` is not a valid regular expression.
    ///
    /// See [Regex syntax](https://docs.rs/regex/latest/regex/#syntax) and
    /// [replacement string syntax](https://docs.rs/regex/latest/regex/struct.Regex.html#replacement-string-syntax).
    pub fn with_key_remap(mut self, pattern: &str, replacement: &str) -> Self {
        let regex = Regex::new(pattern).expect("Invalid regex pattern provided");
        self.key_remap.push((regex, replacement.to_string()));
        self
    }

    /// Enables printing of debug information during loading.
    pub fn with_debug_print(mut self) -> Self {
        self.debug = true;
        self
    }

    /// Sets the adapter type to use for loading tensors.
    pub fn with_adapter_type(mut self, adapter_type: AdapterType) -> Self {
        self.adapter_type = adapter_type;
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

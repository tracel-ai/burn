use std::path::Path;

use burn_store::pytorch::PytorchReader;
use serde::de::DeserializeOwned;

use super::reader::Error;

/// Loads configuration data from a PyTorch `.pth` file.
///
/// This function reads specific configuration or metadata stored in PyTorch checkpoint files.
/// It's particularly useful for extracting model configurations that might be saved alongside
/// the model weights.
///
/// # Arguments
///
/// * `file` - Path to the PyTorch `.pth` file.
/// * `key` - Optional key to filter specific data within the pickle file.
///   If `None`, the entire content is deserialized.
///
/// # Type Parameters
///
/// * `D` - The target type to deserialize into. Must implement `DeserializeOwned`.
///
/// # Returns
///
/// A `Result` containing the deserialized configuration data, or an `Error` if
/// reading or deserialization fails.
///
/// # Examples
///
/// ```ignore
/// use burn_import::pytorch::config::load_config_from_file;
/// use serde::Deserialize;
///
/// #[derive(Debug, Deserialize)]
/// struct ModelConfig {
///     hidden_size: usize,
///     num_layers: usize,
///     // ... other configuration fields
/// }
///
/// let config: ModelConfig = load_config_from_file("model.pth", Some("config"))?;
/// ```
pub fn load_config_from_file<D, P>(file: P, key: Option<&str>) -> Result<D, Error>
where
    D: DeserializeOwned,
    P: AsRef<Path>,
{
    // Use burn-store's PytorchReader to load and deserialize config
    PytorchReader::load_config(file, key).map_err(Error::Store)
}

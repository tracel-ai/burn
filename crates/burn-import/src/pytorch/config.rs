use std::collections::HashMap;
use std::path::Path;

use burn::record::serde::{adapter::DefaultAdapter, data::NestedValue, de::Deserializer};
use burn_store::pytorch::{PytorchReader, PickleValue};
use serde::de::DeserializeOwned;

use super::reader::Error;

/// Reads data from a `.pth` file, specifically looking for configuration data.
///
/// Extracts the pickled data, potentially filtering by a top-level key,
/// and converts it into a `NestedValue`.
///
/// # Arguments
///
/// * `file_path` - The path to the `.pth` file.
/// * `key` - An optional key to select a specific dictionary entry.
///   If `None`, the entire content is returned.
///
/// # Returns
///
/// A `Result` containing the `NestedValue` representation of the data, or an `Error` if
/// reading, parsing, or key extraction fails.
fn read_pt_info<P: AsRef<Path>>(file_path: P, key: Option<&str>) -> Result<NestedValue, Error> {
    // Use the new PytorchReader API to read pickle data
    let pickle_value = PytorchReader::read_pickle_data(file_path, key)?;

    // Convert PickleValue to NestedValue
    convert_pickle_to_nested_value(pickle_value)
}

/// Convert PickleValue to NestedValue
fn convert_pickle_to_nested_value(value: PickleValue) -> Result<NestedValue, Error> {
    Ok(match value {
        PickleValue::None => NestedValue::Default(None),
        PickleValue::Bool(b) => NestedValue::Bool(b),
        PickleValue::Int(i) => NestedValue::I64(i),
        PickleValue::Float(f) => NestedValue::F64(f),
        PickleValue::String(s) => NestedValue::String(s),
        PickleValue::List(list) => {
            let mut vec = Vec::new();
            for item in list {
                vec.push(convert_pickle_to_nested_value(item)?);
            }
            NestedValue::Vec(vec)
        }
        PickleValue::Dict(dict) => {
            let mut map = HashMap::new();
            for (k, v) in dict {
                map.insert(k, convert_pickle_to_nested_value(v)?);
            }
            NestedValue::Map(map)
        }
        PickleValue::Bytes(data) => {
            // Convert bytes to a list of u8 values
            let vec: Vec<NestedValue> = data.into_iter()
                .map(|b| NestedValue::U8(b))
                .collect();
            NestedValue::Vec(vec)
        }
    })
}

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
    // Read the PyTorch file and extract the nested value data
    let nested_value = read_pt_info(file, key)?;

    // Create a deserializer with the default adapter and nested value
    let deserializer = Deserializer::<DefaultAdapter>::new(nested_value, false);

    // Deserialize the nested value into the target type
    let value = D::deserialize(deserializer)?;
    Ok(value)
}

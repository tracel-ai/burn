use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use burn::record::serde::{adapter::DefaultAdapter, data::NestedValue, de::Deserializer};
use burn_store::pytorch::{Object, read_pickle};
use serde::de::DeserializeOwned;
use zip::ZipArchive;

use super::reader::Error;

/// Reads data from a `.pth` file, specifically looking for `data.pkl`.
///
/// Extracts the pickled data, potentially filtering by a top-level key,
/// and converts it into a `NestedValue`.
///
/// # Arguments
///
/// * `file_path` - The path to the `.pth` file.
/// * `key` - An optional key to select a specific dictionary entry within the `data.pkl`.
///   If `None`, the entire content is returned.
///
/// # Returns
///
/// A `Result` containing the `NestedValue` representation of the data, or an `Error` if
/// reading, parsing, or key extraction fails.
fn read_pt_info<P: AsRef<Path>>(file_path: P, key: Option<&str>) -> Result<NestedValue, Error> {
    let mut zip = ZipArchive::new(BufReader::new(File::open(file_path.as_ref())?))?;

    // Find and read the data.pkl file
    let mut pickle_data = Vec::new();
    let mut pickle_found = false;

    // Try standard locations first
    for pickle_path in &["data.pkl", "archive/data.pkl"] {
        if let Ok(mut pickle_file) = zip.by_name(pickle_path) {
            pickle_file.read_to_end(&mut pickle_data)?;
            pickle_found = true;
            break;
        }
    }

    // If not found, search for any file ending with data.pkl
    if !pickle_found {
        for i in 0..zip.len() {
            let file = zip.by_index(i)?;
            let name = file.name().to_string();
            drop(file);

            if name.ends_with("data.pkl") {
                let mut file = zip.by_index(i)?;
                file.read_to_end(&mut pickle_data)?;
                pickle_found = true;
                break;
            }
        }
    }

    if !pickle_found {
        return Err(Error::Other(
            "No data.pkl file found in archive".to_string(),
        ));
    }

    // Parse the pickle data
    let mut reader = BufReader::new(pickle_data.as_slice());
    let obj = read_pickle(&mut reader)?;

    // Convert Object to NestedValue
    let nested_value = convert_object_to_nested_value(obj, key)?;
    Ok(nested_value)
}

/// Convert burn-store's Object to NestedValue
fn convert_object_to_nested_value(obj: Object, key: Option<&str>) -> Result<NestedValue, Error> {
    match obj {
        Object::Dict(dict) => {
            if let Some(key) = key {
                // Extract specific key if requested and return its value directly
                match dict.get(key) {
                    Some(value) => object_to_nested_value(value.clone()),
                    None => Err(Error::Other(format!("Key '{}' not found", key))),
                }
            } else {
                // Convert HashMap<String, Object> to HashMap<String, NestedValue>
                let mut nested_map = HashMap::new();
                for (k, v) in dict {
                    nested_map.insert(k, object_to_nested_value(v)?);
                }
                Ok(NestedValue::Map(nested_map))
            }
        }
        _ => object_to_nested_value(obj),
    }
}

/// Convert a single Object to NestedValue
fn object_to_nested_value(obj: Object) -> Result<NestedValue, Error> {
    match obj {
        Object::String(s) => Ok(NestedValue::String(s)),
        Object::Int(i) => Ok(NestedValue::I64(i)),
        Object::Float(f) => Ok(NestedValue::F64(f)),
        Object::Bool(b) => Ok(NestedValue::Bool(b)),
        Object::None => Ok(NestedValue::Default(None)),
        Object::List(list) => {
            let mut vec = Vec::new();
            for item in list {
                vec.push(object_to_nested_value(item)?);
            }
            Ok(NestedValue::Vec(vec))
        }
        Object::Tuple(tuple) => {
            let mut vec = Vec::new();
            for item in tuple {
                vec.push(object_to_nested_value(item)?);
            }
            Ok(NestedValue::Vec(vec))
        }
        Object::Dict(dict) => {
            let mut map = HashMap::new();
            for (k, v) in dict {
                map.insert(k, object_to_nested_value(v)?);
            }
            Ok(NestedValue::Map(map))
        }
        _ => Err(Error::Other(format!("Unsupported object type: {:?}", obj))),
    }
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

/// Error conversion for zip errors
impl From<zip::result::ZipError> for Error {
    fn from(e: zip::result::ZipError) -> Self {
        Error::Other(format!("Zip error: {}", e))
    }
}

/// Error conversion for IO errors
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Other(format!("IO error: {}", e))
    }
}

/// Error conversion for pickle errors
impl From<burn_store::pytorch::PickleError> for Error {
    fn from(e: burn_store::pytorch::PickleError) -> Self {
        Error::Other(format!("Pickle error: {}", e))
    }
}

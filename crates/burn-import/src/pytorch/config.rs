use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use burn::record::serde::{adapter::DefaultAdapter, data::NestedValue, de::Deserializer};
use candle_core::pickle::{Object, Stack};
use serde::de::DeserializeOwned;
use zip::ZipArchive;

use crate::common::candle::Error;

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
    let mut zip = ZipArchive::new(BufReader::new(File::open(file_path)?))?;

    // We cannot use `zip.by_name` here because we need to find data.pkl in a sub-directory.
    let data_pkl_path = (0..zip.len()).find_map(|i| {
        let file = zip.by_index(i).ok()?; // Use ok() to convert Result to Option
        if file.name().ends_with("data.pkl") {
            Some(file.name().to_string())
        } else {
            None
        }
    });

    let data_pkl_path =
        data_pkl_path.ok_or_else(|| Error::Other("data.pkl not found in archive".to_string()))?;

    let reader = zip.by_name(&data_pkl_path)?;
    let mut reader = BufReader::new(reader);
    let mut stack = Stack::empty();
    stack.read_loop(&mut reader)?;
    let obj = stack.finalize()?;
    let obj = extract_relevant_object(obj, key)?;

    // Convert the PyTorch object to a nested value recursively
    to_nested_value(obj)
}

/// Recursively converts a candle `Object` to a burn `NestedValue`.
///
/// This handles basic types (bool, int, float, string), lists, and dictionaries
/// with string keys.
///
/// # Arguments
///
/// * `obj` - The candle `Object` to convert.
///
/// # Returns
///
/// A `Result` containing the corresponding `NestedValue`, or an `Error` if an
/// unsupported `Object` type is encountered.
fn to_nested_value(obj: Object) -> Result<NestedValue, Error> {
    match obj {
        Object::Bool(v) => Ok(NestedValue::Bool(v)),
        Object::Int(v) => Ok(NestedValue::I32(v)), // Note: Potential truncation from i64
        Object::Float(v) => Ok(NestedValue::F64(v)),
        Object::Unicode(v) => Ok(NestedValue::String(v)),
        Object::List(v) => {
            let list = v
                .into_iter()
                .map(to_nested_value)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(NestedValue::Vec(list))
        }
        Object::Dict(key_values) => {
            let map = key_values
                .into_iter()
                .filter_map(|(name, value)| {
                    // Only keep entries where the key is a Unicode string
                    if let Object::Unicode(name) = name {
                        to_nested_value(value).ok().map(|nv| (name, nv))
                    } else {
                        None
                    }
                })
                .collect::<HashMap<_, _>>();
            Ok(NestedValue::Map(map))
        }
        // Other Object types (e.g., Tuple, Tensor, Storage) are not supported here.
        _ => Err(Error::Other(format!("Unsupported object type: {:?}", obj))),
    }
}

/// Extracts a sub-object from a candle `Object` based on an optional key.
///
/// If a `key` is provided and the `obj` is a `Object::Dict`, it attempts to find
/// the value associated with that key. If the key is not found or `obj` is not a
/// dictionary, an error is returned. If `key` is `None`, the original `obj` is returned.
///
/// # Arguments
///
/// * `obj` - The candle `Object` (expected to be a `Dict` if `key` is `Some`).
/// * `key` - The optional string key to look up in the dictionary.
///
/// # Returns
///
/// A `Result` containing the extracted `Object` or the original `obj`. Returns an `Error`
/// if the key is specified but not found, or if the `obj` is not a dictionary when a key is provided.
fn extract_relevant_object(obj: Object, key: Option<&str>) -> Result<Object, Error> {
    match key {
        Some(key_to_find) => match obj {
            Object::Dict(key_values) => key_values
                .into_iter()
                .find(|(k, _)| match k {
                    Object::Unicode(k_str) => k_str == key_to_find,
                    _ => false,
                })
                .map(|(_, v)| v) // Return the value associated with the key
                .ok_or_else(|| {
                    Error::Other(format!("Key `{key_to_find}` not found in dictionary"))
                }),
            _ => Err(Error::Other(
                "A key was provided, but the loaded object is not a dictionary.".into(),
            )),
        },
        None => Ok(obj), // No key specified, return the whole object
    }
}

/// Deserializes configuration data from a `.pth` file into a specified type `D`.
///
/// Reads the `data.pkl` from the `.pth` archive, potentially extracts data under
/// `top_level_key`, and then deserializes it using burn's serde mechanism.
///
/// # Type Parameters
///
/// * `D` - The target type to deserialize into. Must implement `DeserializeOwned`.
/// * `P` - The type of the path, must implement `AsRef<Path>`.
///
/// # Arguments
///
/// * `path` - The path to the `.pth` file.
/// * `top_level_key` - An optional key to select a specific dictionary entry within
///   the `data.pkl` before deserialization. If `None`, the entire content is deserialized.
///
/// # Returns
///
/// A `Result` containing an instance of type `D`, or an `Error` if file reading,
/// parsing, or deserialization fails.
pub fn config_from_file<D, P>(path: P, top_level_key: Option<&str>) -> Result<D, Error>
where
    D: DeserializeOwned,
    P: AsRef<Path>,
{
    // Read the nested value from the file
    let nested_value = read_pt_info(path, top_level_key)?;

    // Create a deserializer with PyTorch adapter and nested value
    let deserializer = Deserializer::<DefaultAdapter>::new(nested_value, true);

    // Deserialize the nested value into a target type
    let value = D::deserialize(deserializer)?;
    Ok(value)
}

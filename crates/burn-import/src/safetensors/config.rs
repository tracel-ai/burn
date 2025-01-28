use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use super::error::Error;

use burn::record::serde::{adapter::DefaultAdapter, data::NestedValue, de::Deserializer};
use candle_core::pickle::{Object, Stack};
use serde::de::DeserializeOwned;
use zip::ZipArchive;

/// Extracts data from a `.safetensors` file, specifically looking for "data.pkl".
///
/// # Arguments
/// * `file_path` - The path to the `.safetensors` file.
/// * `key` - Optional key to retrieve specific data from the pth file.
///
/// # Returns
///
/// The nested value that can be deserialized into a specific type.
fn read_st_info<P: AsRef<Path>>(file_path: P) -> Result<NestedValue, Error> {
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

    // Convert the PyTorch object to a nested value recursively
    to_nested_value(obj)
}

/// Convert a PyTorch object to a nested value recursively.
///
/// # Arguments
/// * `obj` - The PyTorch object to convert.
///
/// # Returns
/// The nested value.
fn to_nested_value(obj: Object) -> Result<NestedValue, Error> {
    match obj {
        Object::Bool(v) => Ok(NestedValue::Bool(v)),
        Object::Int(v) => Ok(NestedValue::I32(v)),
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
                    if let Object::Unicode(name) = name {
                        let nested_value = to_nested_value(value).ok()?;
                        Some((name, nested_value))
                    } else {
                        None // Skip non-unicode names
                    }
                })
                .collect::<HashMap<_, _>>();
            Ok(NestedValue::Map(map))
        }
        _ => Err(Error::Other("Unsupported value type".into())),
    }
}

/// Deserialize config values  from a `.safetensors` file.
///
/// # Arguments
///
/// * `path` - The path to the `.safetensors` file.
pub fn config_from_file<D, P>(path: P) -> Result<D, Error>
where
    D: DeserializeOwned,
    P: AsRef<Path>,
{
    // Read the nested value from the file
    let nested_value = read_st_info(path)?;

    // Create a deserializer with PyTorch adapter and nested value
    let deserializer = Deserializer::<DefaultAdapter>::new(nested_value, true);

    // Deserialize the nested value into a target type
    let value = D::deserialize(deserializer)?;
    Ok(value)
}

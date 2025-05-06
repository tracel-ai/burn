use std::collections::HashMap;
use std::path::Path;

use crate::common::{
    adapter::PyTorchAdapter,
    candle::{CandleTensor, Error, print_debug_info},
};

use burn::record::PrecisionSettings;
use burn::{
    record::serde::{
        data::{remap, unflatten},
        de::Deserializer,
    },
    tensor::backend::Backend,
};

use candle_core::pickle;
use regex::Regex;
use serde::de::DeserializeOwned;

/// Deserializes tensor data from a PyTorch file (`.pt` or `.pth`) into a Burn record.
///
/// This function reads tensors from a pickle file, optionally remaps their keys,
/// and then deserializes them into the specified record type `D`.
///
/// # Arguments
///
/// * `path` - The path to the PyTorch file to load.
/// * `key_remap` - A list of rules for renaming tensor keys. Each rule is a tuple
///   containing a regular expression to match the original key and a replacement string.
/// * `top_level_key` - An optional key within the pickle file if the tensors are nested
///   under a specific dictionary key (e.g., "state_dict").
/// * `debug` - If `true`, prints information about the loaded tensors and remapped keys.
///
/// # Type Parameters
///
/// * `PS` - The precision settings to use during deserialization.
/// * `D` - The target Burn record type to deserialize into.
/// * `B` - The backend to use for tensor operations (primarily for type context).
///
/// # Returns
///
/// A `Result` containing the deserialized record `D` on success, or an `Error` if
/// reading, remapping, or deserialization fails.
pub fn from_file<PS, D, B>(
    path: &Path,
    key_remap: Vec<(Regex, String)>,
    top_level_key: Option<&str>,
    debug: bool,
) -> Result<D, Error>
where
    D: DeserializeOwned,
    PS: PrecisionSettings,
    B: Backend,
{
    // Read the pickle file and return a map of names to Candle tensors
    let tensors: HashMap<String, CandleTensor> = pickle::read_all_with_key(path, top_level_key)?
        .into_iter()
        .map(|(key, tensor)| (key, CandleTensor(tensor)))
        .collect();

    // Remap the tensor keys based on the provided rules
    let (tensors, remapped_keys) = remap(tensors, key_remap);

    // Print debug information if enabled
    if debug {
        print_debug_info(&tensors, remapped_keys);
    }

    // Convert the flat map of tensors into a nested data structure suitable for deserialization
    let nested_value = unflatten::<PS, _>(tensors)?;

    // Create a deserializer using the PyTorch adapter and the nested tensor data
    let deserializer = Deserializer::<PyTorchAdapter<PS, B>>::new(nested_value, true);

    // Deserialize the nested data structure into the target record type
    let value = D::deserialize(deserializer)?;
    Ok(value)
}

use std::{collections::HashMap, path::Path};

use burn::{
    record::{
        PrecisionSettings,
        serde::{
            adapter::DefaultAdapter,
            data::{remap, unflatten},
            de::Deserializer,
        },
    },
    tensor::backend::Backend,
};

use candle_core::{Device, safetensors};
use regex::Regex;
use serde::de::DeserializeOwned;

use super::super::common::adapter::PyTorchAdapter;
use super::recorder::AdapterType;
use crate::common::candle::{CandleTensor, Error, print_debug_info};

/// Deserializes model state from a safetensors file.
///
/// # Arguments
///
/// * `path` - Path to the safetensors file.
/// * `key_remap` - A vector of tuples containing regular expressions and replacement strings
///   for remapping tensor keys.
/// * `debug` - If true, prints debug information about the loaded tensors and remapped keys.
/// * `adapter_type` - Specifies the adapter to use for deserialization (e.g., PyTorch, None).
pub fn from_file<PS, D, B>(
    path: &Path,
    key_remap: Vec<(Regex, String)>,
    debug: bool,
    adapter_type: AdapterType,
) -> Result<D, Error>
where
    D: DeserializeOwned,
    PS: PrecisionSettings,
    B: Backend,
{
    // Load tensors from the safetensors file into a HashMap.
    let tensors: HashMap<String, CandleTensor> = safetensors::load(path, &Device::Cpu)?
        .into_iter()
        .map(|(key, tensor)| (key, CandleTensor(tensor)))
        .collect();

    // Remap tensor keys based on the provided patterns.
    let (tensors, remapped_keys) = remap(tensors, key_remap);

    // Optionally print debug information about tensors and key remapping.
    if debug {
        print_debug_info(&tensors, remapped_keys);
    }

    // Convert the flat map of tensors into a nested data structure suitable for deserialization.
    let nested_value = unflatten::<PS, _>(tensors)?;

    // Deserialize the nested data structure into the target type using the specified adapter.
    let value = match adapter_type {
        AdapterType::PyTorch => D::deserialize(Deserializer::<PyTorchAdapter<PS, B>>::new(
            nested_value,
            true, // Allow unexpected fields by default? Might need clarification.
        ))?,
        AdapterType::NoAdapter => {
            D::deserialize(Deserializer::<DefaultAdapter>::new(nested_value, true))?
        }
        AdapterType::TensorFlow => {
            todo!("TensorFlow adapter deserialization is not yet implemented.")
        }
    };

    Ok(value)
}

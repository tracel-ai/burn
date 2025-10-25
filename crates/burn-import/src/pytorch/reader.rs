use std::collections::HashMap;
use std::path::Path;

use crate::common::{
    adapter::PyTorchAdapter,
    tensor_snapshot::{TensorSnapshotWrapper, print_debug_info},
};

use burn::record::PrecisionSettings;
use burn::{
    record::serde::{
        data::{remap, unflatten},
        de::Deserializer,
    },
    tensor::backend::Backend,
};

use burn_store::pytorch::PytorchReader;
use regex::Regex;
use serde::de::DeserializeOwned;

/// Error type for PyTorch file operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Store error: {0}")]
    Store(#[from] burn_store::pytorch::PytorchError),

    #[error("Serde error: {0}")]
    Serde(#[from] burn::record::serde::error::Error),

    #[error("Other error: {0}")]
    Other(String),
}

/// Deserializes tensor data from a PyTorch file (`.pt` or `.pth`) into a Burn record.
///
/// This function reads tensors from a pickle file using burn-store's PyTorch reader,
/// optionally remaps their keys, and then deserializes them into the specified record type `D`.
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
    // Use burn-store's PyTorch reader to load tensors
    let reader = if let Some(key) = top_level_key {
        PytorchReader::with_top_level_key(path, key)?
    } else {
        PytorchReader::new(path)?
    };

    // Get the tensors as TensorSnapshots and wrap them
    let tensors: HashMap<String, TensorSnapshotWrapper> = reader
        .into_tensors()
        .into_iter()
        .map(|(key, snapshot)| (key, TensorSnapshotWrapper(snapshot)))
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

// Re-export burn-store's PyTorch reader types for convenience

// Implement conversion to RecorderError for compatibility with the Recorder trait
impl From<Error> for burn::record::RecorderError {
    fn from(error: Error) -> Self {
        burn::record::RecorderError::DeserializeError(error.to_string())
    }
}

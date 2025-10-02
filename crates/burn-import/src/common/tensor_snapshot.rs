//! TensorSnapshot support for burn-import.

use burn::record::PrecisionSettings;
use burn::record::serde::{
    data::{NestedValue, Serializable},
    error,
    ser::Serializer,
};
use burn_store::TensorSnapshot;
use serde::Serialize;
use std::collections::HashMap;
use std::ops::Deref;

/// Wrapper for TensorSnapshot to implement Serializable
pub struct TensorSnapshotWrapper(pub TensorSnapshot);

impl Deref for TensorSnapshotWrapper {
    type Target = TensorSnapshot;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Serializes a TensorSnapshot.
///
/// Tensors are wrapped in a `Param` struct (learnable parameters) and serialized as a `TensorData` struct.
impl Serializable for TensorSnapshotWrapper {
    fn serialize<PS>(&self, serializer: Serializer) -> Result<NestedValue, error::Error>
    where
        PS: PrecisionSettings,
    {
        // Get the tensor data
        let data = self
            .0
            .to_data()
            .map_err(|e| error::Error::Other(format!("Failed to get tensor data: {:?}", e)))?;
        let shape = data.shape.clone();
        let dtype = data.dtype;
        let bytes = data.into_bytes();

        // Create the tensor data structure
        let mut tensor_data: HashMap<String, NestedValue> = HashMap::new();
        tensor_data.insert("bytes".into(), NestedValue::Bytes(bytes));
        tensor_data.insert("shape".into(), shape.serialize(serializer.clone())?);
        tensor_data.insert("dtype".into(), dtype.serialize(serializer)?);

        // Create the param structure
        let param_id = self.0.tensor_id.unwrap_or_default();
        let mut param: HashMap<String, NestedValue> = HashMap::new();
        param.insert("id".into(), NestedValue::String(param_id.serialize()));
        param.insert("param".into(), NestedValue::Map(tensor_data));

        Ok(NestedValue::Map(param))
    }
}

/// Print debug information about tensors
pub fn print_debug_info(
    tensors: &HashMap<String, TensorSnapshotWrapper>,
    remapped_keys: Vec<(String, String)>,
) {
    let mut remapped_keys = remapped_keys;
    remapped_keys.sort();
    println!("Debug information of keys and tensor shapes:\n---");
    for (new_key, old_key) in remapped_keys {
        if old_key != new_key {
            println!("Original Key: {old_key}");
            println!("Remapped Key: {new_key}");
        } else {
            println!("Key: {new_key}");
        }

        let snapshot = &tensors[&new_key].0;
        let shape = &snapshot.shape;
        let dtype = &snapshot.dtype;
        println!("Shape: {shape:?}");
        println!("Dtype: {dtype:?}");
        println!("---");
    }
}

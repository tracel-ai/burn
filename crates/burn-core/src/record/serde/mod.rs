//! Module contains the serde implementation for the record module
//! useful for custom importing model weights, such as PyTorch's pt file format.

/// The adapter trait that is used to convert the nested value to the module type.
pub mod adapter;

/// The main data structure used for deserialization.
pub mod data;

/// The deserializer that is used to convert the nested value to the record.
pub mod ser;

/// The deserializer that is used to convert the nested value to the record.
pub mod de;

/// Error types.
pub mod error;

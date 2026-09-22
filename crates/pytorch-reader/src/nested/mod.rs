//! Serde-based deserialization of nested values, which is how
//! [`PytorchReader::load_config`](crate::PytorchReader::load_config) turns the non-tensor
//! part of a checkpoint into a typed value.

/// The adapter trait that is used to convert the nested value to the module type.
pub mod adapter;

/// The main data structure used for deserialization.
pub mod data;

/// The deserializer that converts a nested value into a typed item.
pub mod de;

/// Error types.
pub mod error;

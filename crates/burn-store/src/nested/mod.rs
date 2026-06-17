//! Serde-based deserialization of nested values, used for importing model weights from external
//! formats (e.g. PyTorch's pickle `.pt` files) into Burn modules via `burn-store`.

/// The adapter trait that is used to convert the nested value to the module type.
pub mod adapter;

/// The main data structure used for deserialization.
pub mod data;

/// The deserializer that converts a nested value into a typed item.
pub mod de;

/// Error types.
pub mod error;

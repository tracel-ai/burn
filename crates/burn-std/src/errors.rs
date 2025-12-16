//! # Common Burn Errors

use alloc::string::String;

/// Access Bounds Error.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct BoundsError {
    /// The name/type of the index.
    pub index_name: String,
    /// The index value.
    pub index: String,
    /// The name/type of the bounds.
    pub bounds_name: String,
    /// The bounds value.
    pub bounds: String,
}

impl core::fmt::Display for BoundsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} ({}) out of bounds for {} ({})",
            self.index_name, self.index, self.bounds_name, self.bounds
        )
    }
}

impl core::error::Error for BoundsError {}

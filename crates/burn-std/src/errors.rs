//! # Common Burn Errors

use alloc::string::String;
use core::ops::Range;

/// Describes the kind of an index.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum IndexKind {
    /// The index of an element in a dimension.
    Element,

    /// The index of a dimension.
    Dimension,
}

impl IndexKind {
    /// Get the display name of the kind.
    pub fn name(&self) -> &'static str {
        match self {
            IndexKind::Element => "element",
            IndexKind::Dimension => "dimension",
        }
    }
}

/// Access Bounds Error.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum BoundsError {
    /// Generic bounds error.
    Generic(String),

    /// Index out of bounds.
    Index {
        /// The kind of index that was out of bounds.
        kind: IndexKind,

        /// The index that was out of bounds.
        index: isize,

        /// The range of valid indices.
        bounds: Range<isize>,
    },
}

impl BoundsError {
    /// Create a new index error.
    pub fn index(kind: IndexKind, index: isize, bounds: Range<isize>) -> Self {
        Self::Index {
            kind,
            index,
            bounds,
        }
    }
}

impl core::fmt::Display for BoundsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Generic(msg) => write!(f, "BoundsError: {}", msg),
            Self::Index {
                kind,
                index,
                bounds: range,
            } => write!(
                f,
                "BoundsError: {} {} out of bounds: {:?}",
                kind.name(),
                index,
                range
            ),
        }
    }
}

impl core::error::Error for BoundsError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::ToString;

    #[test]
    fn test_bounds_error_display() {
        assert_eq!(
            format!("{}", BoundsError::Generic("test".to_string())),
            "BoundsError: test"
        );
        assert_eq!(
            format!(
                "{}",
                BoundsError::Index {
                    kind: IndexKind::Element,
                    index: 1,
                    bounds: 0..2
                }
            ),
            "BoundsError: element 1 out of bounds: 0..2"
        );
    }
}

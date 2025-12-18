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

/// Common Expression Error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionError {
    /// Parse Error.
    ParseError {
        /// The error message.
        message: String,
        /// The source expression.
        source: String,
    },

    /// Invalid Expression.
    InvalidExpression {
        /// The error message.
        message: String,
        /// The source expression.
        source: String,
    },
}

impl core::fmt::Display for ExpressionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ParseError { message, source } => {
                write!(f, "ExpressionError: ParseError: {} ({})", message, source)
            }
            Self::InvalidExpression { message, source } => write!(
                f,
                "ExpressionError: InvalidExpression: {} ({})",
                message, source
            ),
        }
    }
}

impl core::error::Error for ExpressionError {}

impl ExpressionError {
    /// Constructs a new [`ExpressionError::ParseError`].
    ///
    /// This function is a utility for creating instances where a parsing error needs to be represented,
    /// encapsulating a descriptive error message and the source of the error.
    ///
    /// # Parameters
    ///
    /// - `message`: A value that can be converted into a `String`, representing a human-readable description
    ///   of the parsing error.
    /// - `source`: A value that can be converted into a `String`, typically identifying the origin or
    ///   input that caused the parsing error.
    pub fn parse_error(message: impl Into<String>, source: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
            source: source.into(),
        }
    }

    /// Creates a new [`ExpressionError::InvalidExpression`].
    ///
    /// # Parameters
    /// - `message`: A detailed message describing the nature of the invalid expression.
    ///   Accepts any type that can be converted into a `String`.
    /// - `source`: The source or context in which the invalid expression occurred.
    ///   Accepts any type that can be converted into a `String`.
    pub fn invalid_expression(message: impl Into<String>, source: impl Into<String>) -> Self {
        Self::InvalidExpression {
            message: message.into(),
            source: source.into(),
        }
    }
}
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

    #[test]
    fn test_parse_error() {
        let err = ExpressionError::parse_error("test", "source");
        assert_eq!(
            format!("{:?}", err),
            "ParseError { message: \"test\", source: \"source\" }"
        );
    }

    #[test]
    fn test_invalid_expression() {
        let err = ExpressionError::invalid_expression("test", "source");
        assert_eq!(
            format!("{:?}", err),
            "InvalidExpression { message: \"test\", source: \"source\" }"
        );
    }
}

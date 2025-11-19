//! This module defines common structures and methods to handle parsing errors
//! and expression-related error handling. It provides an extensible way
//! to encapsulate and manage error metadata for usage in parsing or evaluating expressions.

use alloc::string::String;

/// Common Parse Error.
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

impl ExpressionError {
    /// Constructs a new `ParseError`.
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

    /// Creates a new `InvalidExpression`.
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

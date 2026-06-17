/// The error type for nested-value serde.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Failed to deserialize.
    #[error("failed to deserialize: {0}")]
    Deserialize(#[from] serde::de::value::Error),

    /// Failed to serialize.
    #[error("failed to serialize")]
    Serialize(String),

    /// Encountered an invalid state.
    #[error("invalid state")]
    InvalidState,

    /// Other error.
    #[error("other error: {0}")]
    Other(String),
}

impl serde::de::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error::Deserialize(serde::de::value::Error::custom(msg.to_string()))
    }
}

impl serde::ser::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error::Serialize(msg.to_string())
    }
}

use std::path::PathBuf;

use burn::record::RecorderError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to read file at {0}")]
    ReadFile(PathBuf, #[source] std::io::Error),

    #[error("failed to deserialize")]
    Deserialize(#[from] serde::de::value::Error),

    #[error("failed to serialize")]
    Serialize(String),

    #[error("invalid state")]
    InvalidState,

    // Add other kinds of errors as needed
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

// Implement From trait for Error to RecorderError
impl From<Error> for RecorderError {
    fn from(error: Error) -> Self {
        RecorderError::DeserializeError(error.to_string())
    }
}

use std::path::PathBuf;

use burn::record::RecorderError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to read file at {0}")]
    ReadFileError(PathBuf, #[source] std::io::Error),

    #[error("failed to deserialize")]
    DeserializeError(#[from] serde::de::value::Error),

    #[error("failed to serialize")]
    SerializeError(String),

    // Add other kinds of errors as needed
    #[error("other error")]
    Other,
}

impl serde::de::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error::DeserializeError(serde::de::value::Error::custom(msg.to_string()))
    }
}

impl serde::ser::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error::SerializeError(msg.to_string())
    }
}

// Implement From trait for Error to RecorderError
impl From<Error> for RecorderError {
    fn from(error: Error) -> Self {
        RecorderError::DeserializeError(error.to_string())
    }
}

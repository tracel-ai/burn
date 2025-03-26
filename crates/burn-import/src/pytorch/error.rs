use burn::record::{RecorderError, serde::error};
use zip::result::ZipError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Serde error: {0}")]
    Serde(#[from] error::Error),

    #[error("Candle pickle error: {0}")]
    CandlePickle(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Zip error: {0}")]
    Zip(#[from] ZipError),

    // Add other kinds of errors as needed
    #[error("other error: {0}")]
    Other(String),
}

// Implement From trait for Error to RecorderError
impl From<Error> for RecorderError {
    fn from(error: Error) -> Self {
        RecorderError::DeserializeError(error.to_string())
    }
}

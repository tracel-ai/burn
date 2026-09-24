use std::fmt;

/// Default error type used by [`Dataset`](crate::Dataset) implementations that don't define
/// their own.
///
/// Wraps specific dataset error types (e.g. `SqliteDatasetError`) at call sites that don't need
/// the concrete error type. Use `result.map_err(DatasetError::new)?` to convert and propagate an
/// error; `?` alone does not perform this wrapping.
#[derive(Debug)]
pub struct DatasetError(Box<dyn std::error::Error + Send + Sync + 'static>);

impl DatasetError {
    /// Wraps an arbitrary error as a [`DatasetError`].
    pub fn new<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self(Box::new(err))
    }
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

use std::fmt;

/// Default error type used by [`Dataset`](crate::Dataset) implementations that don't define
/// their own.
///
/// Wraps any error so that specific dataset error types (e.g. `SqliteDatasetError`) can be
/// converted into it with `?` at call sites that don't care about the concrete error.
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

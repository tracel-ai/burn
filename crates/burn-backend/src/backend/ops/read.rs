use crate::ExecutionError;

/// A read in flight, resolving to its data or the error that stopped it.
///
/// Every `Send` future of that output is one, so a backend returns whichever it builds.
pub trait ReadFuture<T>: Future<Output = Result<T, ExecutionError>> + Send {}

impl<T, F> ReadFuture<T> for F where F: Future<Output = Result<T, ExecutionError>> + Send {}

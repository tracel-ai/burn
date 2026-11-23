use crate::ExecutionError;

/// A read in flight, resolving to its data or the error that stopped it.
///
/// It yields the tensor as it was at the call, whatever is queued on the device after it, which a
/// remote server relies on to read between two graph replays. Every `Send` future of that output
/// is one, so a backend returns whichever it builds.
pub trait ReadFuture<T>: Future<Output = Result<T, ExecutionError>> + Send {}

impl<T, F> ReadFuture<T> for F where F: Future<Output = Result<T, ExecutionError>> + Send {}

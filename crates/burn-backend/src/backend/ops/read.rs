use crate::ExecutionError;

/// A read in flight, resolving to its data or the error that stopped it.
///
/// Work queued after the call does not change what the read yields. A backend can defer copying
/// while the source storage stays immutable. A graph replay writes its buffers in place, so a
/// remote server queues the copy when it submits the read, before any later replay.
///
/// Any `Send` future with output `Result<T, ExecutionError>` implements this trait.
pub trait ReadFuture<T>: Future<Output = Result<T, ExecutionError>> + Send {}

impl<T, F> ReadFuture<T> for F where F: Future<Output = Result<T, ExecutionError>> + Send {}

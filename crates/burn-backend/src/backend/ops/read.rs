use crate::ExecutionError;

/// A read in flight, resolving to its data or the error that stopped it.
///
/// Work queued after the call does not change what it yields. A graph replay writes its buffers
/// in place, so only a read that copies at the call, as a remote server's does, is safe across one.
/// Every `Send` future of that output is one, so a backend returns whichever it builds.
pub trait ReadFuture<T>: Future<Output = Result<T, ExecutionError>> + Send {}

impl<T, F> ReadFuture<T> for F where F: Future<Output = Result<T, ExecutionError>> + Send {}

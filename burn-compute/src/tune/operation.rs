use alloc::string::String;
use alloc::vec::Vec;

use crate::server::ComputeServer;

/// Type of operation for the kernel
pub trait AutotuneOperationSet<S>: Send {
    /// The key used in the tune cache
    fn key(&self) -> AutotuneKey;

    /// All candidate operations for autotuning this operation type
    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation<S>>>;

    /// Returns the operation for the given index, matching the order
    /// returned by autotunables
    fn fastest(&self, fastest_index: usize) -> Box<dyn AutotuneOperation<S>>;
}

pub trait AutotuneOperation<S: ComputeServer> {
    /// Executes the operation on actual inputs with the fastest kernel
    fn execute(self: Box<Self>);
    // TODO validate that after execute, output should have one ref only

    /// Executes the operation on artificial inputs on all kernels (not sure we keep)
    fn execute_for_autotune(self: Box<Self>);

    fn clone(&self) -> Box<dyn AutotuneOperation<S>>;
}

#[derive(new, Clone, Debug, PartialEq, Eq, Hash)]
/// The key used in the tune cache, referring to the operation type,
/// generally hardcoded for an autotune operation, and to the input shape
pub struct AutotuneKey {
    operation: String,
    input_description: String,
}

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
    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation<S>>;
}

pub trait AutotuneOperation<S: ComputeServer> {
    fn execute(self: Box<Self>);

    fn clone(&self) -> Box<dyn AutotuneOperation<S>>;
}

#[derive(new, Clone, Debug, PartialEq, Eq, Hash)]
/// The key used in the tune cache, referring to the operation type,
/// generally hardcoded for an autotune operation, and to the input shape
pub struct AutotuneKey {
    operation: String,
    input_description: String,
}

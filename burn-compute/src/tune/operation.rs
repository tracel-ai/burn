use alloc::string::String;
use alloc::vec::Vec;

/// Type of operation for the kernel
pub trait AutotuneOperationSet: Send {
    /// The key used in the tune cache
    fn key(&self) -> AutotuneKey;

    /// All candidate operations for autotuning this operation type
    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation>>;

    /// Returns the operation for the given index, matching the order
    /// returned by autotunables
    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation>;
}

pub trait AutotuneOperation {
    fn execute(self: Box<Self>);

    fn clone(&self) -> Box<dyn AutotuneOperation>;
}

#[derive(new, Clone, Debug, PartialEq, Eq, Hash)]
/// The key used in the tune cache, referring to the operation type,
/// generally hardcoded for an autotune operation, and to the input shape
pub struct AutotuneKey {
    operation: String,
    input_description: String,
}

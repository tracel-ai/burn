use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Display;

/// Groups operations of the same type for autotune
pub trait AutotuneOperationSet: Send {
    /// The key used in the tune cache
    fn key(&self) -> AutotuneKey;

    /// All candidate operations for autotuning this operation type
    /// Operations can run on toy tensors of relevant size
    fn autotunables(&self) -> Vec<Box<dyn AutotuneOperation>>;

    /// Returns the operation for the given index, matching the order
    /// returned by autotunables. Operation obtained here runs on original tensors
    fn fastest(self: Box<Self>, fastest_index: usize) -> Box<dyn AutotuneOperation>;
}

/// Contains operation to run and inputs on which to run it
pub trait AutotuneOperation {
    /// Runs the operation
    fn execute(self: Box<Self>);

    /// The name of the operation.
    fn name(&self) -> &str {
        core::any::type_name::<Self>()
    }

    /// Clones the operation and inputs
    fn clone(&self) -> Box<dyn AutotuneOperation>;
}

#[derive(new, Clone, Debug, PartialEq, Eq, Hash)]
/// The key used in the tune cache, referring to the operation type,
/// generally hardcoded for an autotune operation, and to the input shape
pub struct AutotuneKey {
    operation: String,
    input_description: String,
}

impl Display for AutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{}-{}", self.operation, self.input_description).as_str())
    }
}

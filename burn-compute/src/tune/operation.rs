use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::hash::Hash;

/// Groups operations of the same type for autotune
pub trait AutotuneOperationSet<K>: Send {
    /// The key used in the tune cache
    fn key(&self) -> K;

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

/// Trait alias
pub trait AutotuneKey: Clone + Debug + PartialEq + Eq + Hash + Display {}
impl AutotuneKey for String {}

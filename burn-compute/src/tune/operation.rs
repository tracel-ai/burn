use core::hash::Hash;

/// Format of the inputs and outputs of a kernel
pub trait HashableResources: PartialEq + Eq + Hash {
    /// Description used as an autotune key
    fn key(&self) -> String;
}

/// Type of operation for the kernel
pub trait Operation: PartialEq + Eq + Hash {
    /// Input and output format for the operation
    type Resources: HashableResources;
}

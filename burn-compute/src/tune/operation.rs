/// Format of the inputs and outputs of a kernel
pub trait HashableResources {
    /// Description used as an autotune key
    fn key(&self) -> String;
}

/// Type of operation for the kernel
// pub trait Operation: PartialEq + Eq + Hash {
pub trait Operation {
    /// Input and output format for the operation
    type Resources: HashableResources;
}

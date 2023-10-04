use core::hash::Hash;

// Is the same implementation for all kernels of the same kind, like all matmul implementations
// But their hash will be different depending on input size for instance
pub trait TuneKey: Hash {}

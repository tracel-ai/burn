//! PyTorch format support for burn-store.

pub mod pickle_reader;
pub mod reader;

#[cfg(test)]
pub mod tests;

// Main public interface
pub use reader::{PickleValue, PytorchReader};

// Re-export error type for convenience
pub use reader::Error as PytorchError;

//! PyTorch format support for burn-store.

pub mod pickle_reader;

pub use pickle_reader::{Error as PickleError, Object, OpCode, read_pickle, read_pickle_tensors};

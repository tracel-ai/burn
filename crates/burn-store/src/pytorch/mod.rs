//! PyTorch format support for burn-store.

pub mod pickle_reader;
pub mod reader;

pub use pickle_reader::{Error as PickleError, Object, OpCode, read_pickle, read_pickle_tensors};
pub use reader::{
    Error as ReaderError, PytorchReader, load_pytorch_file, read_pytorch_file, read_pytorch_tensors,
};

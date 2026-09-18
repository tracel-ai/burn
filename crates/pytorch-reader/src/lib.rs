//! Read PyTorch checkpoint files (`.pt`, `.pth`) without PyTorch or Burn.
//!
//! A checkpoint is a pickle describing the saved object, with each tensor's bytes kept in a
//! separate storage inside the same container. [`PytorchReader`] parses the pickle and hands
//! back a [`Tensor`] per entry, carrying its name, [`DType`] and shape; the bytes are
//! produced only when [`Tensor::read`] asks for them (from the file for ZIP and legacy
//! containers, from memory for TAR, whose storages are read at open).
//!
//! ```rust,no_run
//! use pytorch_reader::PytorchReader;
//!
//! let reader = PytorchReader::new("model.pt")?;
//! for tensor in reader.tensors().values() {
//!     println!("{}: {:?} {:?}", tensor.name, tensor.dtype(), tensor.shape());
//! }
//!
//! // Bytes are read from the file here, not at open.
//! let weight = reader.get("fc.weight").unwrap();
//! let bytes: Vec<u8> = weight.read()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! A checkpoint that nests its weights under a key (`"state_dict"`, `"model"`, ...) is
//! opened with [`PytorchReader::with_top_level_key`]. Values that are not tensors, such as a
//! configuration dictionary saved next to the weights, can be deserialized into any `serde`
//! type with [`PytorchReader::load_config`].
//!
//! # Supported containers
//!
//! Every layout `torch.save` has written is read, and detected from the file itself.
//!
//! **ZIP (PyTorch 1.6 and later).** An archive holding, under a root directory usually named
//! after the file: `data.pkl` with the pickled tensor metadata, one entry per tensor storage
//! under `data/`, and small text entries such as `version` and `byteorder`.
//!
//! **Legacy pickle stream (PyTorch 0.1.10 through 1.5).** Sequential pickles: a magic number
//! (`0x1950a86a20f9469cfc6c`), the protocol version (1001), system info (endianness, type
//! sizes), the saved object, then the storage key list followed by each storage as an `i64`
//! element count and its bytes.
//!
//! **TAR (before PyTorch 0.1.10, e.g. early torchvision models).** An archive of `sys_info`
//! (a system info pickle), `storages` (a count pickle, then per storage a metadata pickle,
//! element count and bytes), `tensors` (a count pickle, then per tensor a metadata pickle and
//! binary shape, stride and storage offset) and `pickle` (the saved object, referencing
//! tensors by persistent id).
//!
//! **Plain pickle.** A pickle with a dictionary at its root, as some tools write for
//! configuration on its own.
//!
//! Full-model saves (as opposed to a `state_dict`) are refused, as are checkpoints holding
//! sparse, quantized or nested tensors, since those cannot be represented. Only little-endian
//! files are supported.
//!
//! # Safety limits
//!
//! Parsing is hardened against malformed or hostile files. A ZIP checkpoint whose `data.pkl`
//! claims more than [`MAX_PICKLE_SIZE`], or any checkpoint holding a tensor that would
//! materialize more than [`MAX_TENSOR_SIZE`] bytes, is refused at open before an allocation
//! of that size is made, and every length a file declares is checked against what the file
//! actually holds.

pub mod nested;
mod pickle_reader;
mod reader;
mod storage;
mod tensor;

#[cfg(test)]
mod tests;

pub use pickle_reader::{OpCode, PickleError};
pub use reader::{
    ByteOrder, FileFormat, PickleValue, PytorchError, PytorchMetadata, PytorchReader,
};
pub use tensor::{DType, Tensor};

/// Largest `data.pkl` accepted from a ZIP checkpoint (100 MiB).
///
/// Only that container needs a ceiling: a deflated entry can claim any decompressed size,
/// while the other containers' pickles are parsed straight off the file. The pickle holds
/// tensor metadata and whatever non-tensor values were saved alongside; the tensor bytes
/// live in separate entries and are never subject to this limit.
pub const MAX_PICKLE_SIZE: u64 = 100 * 1024 * 1024;

/// Largest tensor accepted: 2 GiB on 32-bit targets, 10 GiB elsewhere.
///
/// A view can declare far more elements than its storage holds (an `expand` has stride 0),
/// so the file itself gives no bound on what a tensor materializes to. A checkpoint holding
/// a tensor whose bytes would exceed this fails to open.
#[cfg(target_pointer_width = "32")]
pub const MAX_TENSOR_SIZE: usize = 2 * 1024 * 1024 * 1024;
#[cfg(not(target_pointer_width = "32"))]
pub const MAX_TENSOR_SIZE: usize = 10 * 1024 * 1024 * 1024;

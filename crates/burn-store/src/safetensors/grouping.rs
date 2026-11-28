//! Reading tensors a safetensors file splits across several entries as one quantized tensor.
//!
//! A release quantized to a block format stores each weight's codes and its block scales as two
//! entries (`<p>.weight` beside `<p>.scale`, `.weight_scale_inv`, `.weight_scale`…), linked only
//! by their names and by the release's own configuration. Nothing in the file says which entries
//! pair up, so the caller says it: a [`TensorGrouping`] sees every entry of a file before any is
//! read and names the groups that form one [`QFloat`](burn_core::tensor::DType::QFloat) tensor.

use alloc::string::String;
use alloc::vec::Vec;
use burn_core::tensor::quantization::QuantScheme;

use super::SafetensorsStoreError;

/// A dtype as a safetensors header names it, before burn reads it as one of its own.
pub use safetensors::Dtype as SafetensorsDtype;

/// One entry of a safetensors file's header: what a [`TensorGrouping`] decides on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsEntry {
    /// The entry's name in the file, before any remapping.
    pub name: String,
    /// The dtype the file stores it at.
    pub dtype: SafetensorsDtype,
    /// Its shape as the file states it.
    pub shape: Vec<usize>,
}

/// Entries of one file read together as a single quantized tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedGroup {
    /// The name the tensor is read under, before any remapping.
    pub name: String,
    /// The entry holding the codes, laid out as `scheme` stores its values.
    pub codes: String,
    /// The entry holding the block scales, one per block in row-major block order, encoded at
    /// `scheme`'s scale dtype.
    pub scales: String,
    /// The entry holding the per-tensor scale, for a two-level `scheme`.
    pub tensor_scale: Option<String>,
    /// The scheme the codes and scales are read under.
    pub scheme: QuantScheme,
    /// The tensor's shape, which a packed store's codes entry does not state.
    pub shape: Vec<usize>,
}

/// Which entries of a safetensors file form quantized tensors.
///
/// The store asks once per file, with every entry of its header, before converting any dtype or
/// renaming any key. An entry a group names is read only through that group; every other entry
/// is read as burn reads it alone, and one whose dtype burn does not hold fails the read.
pub trait TensorGrouping: Send + Sync {
    /// The quantized tensors `entries` hold. Each entry is named by at most one group.
    ///
    /// # Errors
    ///
    /// Whatever the implementation finds wrong with the file, such as codes without their scales.
    fn group(
        &self,
        entries: &[SafetensorsEntry],
    ) -> Result<Vec<QuantizedGroup>, SafetensorsStoreError>;
}

/// The grouping a store starts with: every entry is a tensor of its own.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoGrouping;

impl TensorGrouping for NoGrouping {
    fn group(
        &self,
        _entries: &[SafetensorsEntry],
    ) -> Result<Vec<QuantizedGroup>, SafetensorsStoreError> {
        Ok(Vec::new())
    }
}

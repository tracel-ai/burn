use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use burn::store::RecordError;
use burn::tensor::Bytes;
use burn_core as burn;

use burn_pack::{Reader, Scalar, Writer};

/// A serialized optimizer state, stored in the [burnpack](burn_pack) format.
///
/// Unlike a module record (keyed by module path), an optimizer record is keyed per parameter:
/// each parameter's state is decomposed into tensors named `"{param_id}.{field}"` (carrying the
/// originating `param_id`) plus a few typed scalar entries kept in the burnpack scalar map.
///
/// Obtain one from a [`ModuleOptimizer`](crate::optim::ModuleOptimizer) with
/// [`to_record`](crate::optim::ModuleOptimizer::to_record), then save it
/// ([`save`](Self::save) / [`into_bytes`](Self::into_bytes)) or apply it back with
/// [`load_record`](crate::optim::ModuleOptimizer::load_record).
#[derive(Default)]
pub struct OptimizerRecord {
    pub(crate) tensors: Vec<burn_pack::Tensor>,
    pub(crate) scalars: BTreeMap<String, Scalar>,
}

impl core::fmt::Debug for OptimizerRecord {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OptimizerRecord")
            .field("num_tensors", &self.tensors.len())
            .field("num_scalars", &self.scalars.len())
            .finish()
    }
}

impl OptimizerRecord {
    /// The number of tensors in the record.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the record holds no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Serialize the record to an in-memory burnpack byte buffer.
    pub fn into_bytes(self) -> Result<Bytes, RecordError> {
        Ok(self.into_pack_writer().into_bytes()?)
    }

    /// Reconstruct a record from an in-memory burnpack byte buffer.
    pub fn from_bytes(bytes: Bytes) -> Result<Self, RecordError> {
        Self::from_pack_reader(Reader::from_bytes(bytes)?)
    }

    /// Stream the record to any [`std::io::Write`] without materializing the whole buffer first.
    #[cfg(feature = "std")]
    pub fn into_writer<W: std::io::Write>(self, writer: W) -> Result<(), RecordError> {
        self.into_pack_writer().write_to(writer)?;
        Ok(())
    }

    /// Reconstruct a record by streaming from any [`std::io::Read`].
    #[cfg(feature = "std")]
    pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Self, RecordError> {
        Self::from_pack_reader(Reader::from_reader(reader)?)
    }

    /// Save the record to a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn save<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), RecordError> {
        self.into_pack_writer().write_to_file(path)?;
        Ok(())
    }

    /// Load a record from a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RecordError> {
        Self::from_pack_reader(Reader::from_file(path)?)
    }

    fn into_pack_writer(self) -> Writer {
        let mut writer = Writer::new(self.tensors);
        for (key, value) in &self.scalars {
            writer = writer.with_scalar(key, *value);
        }
        writer
    }

    fn from_pack_reader(reader: Reader) -> Result<Self, RecordError> {
        let scalars = reader.scalars().clone();
        let tensors = reader.into_tensors()?;
        Ok(Self { tensors, scalars })
    }
}

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use burn::store::RecordError;
use burn::tensor::{Bytes, TensorData};
use burn_core as burn;

use burn_pack::{Reader, Scalar, Writer};

/// A single optimizer state tensor and its optional originating parameter id.
pub(crate) struct RecordTensor {
    pub(crate) name: String,
    pub(crate) param_id: Option<u64>,
    pub(crate) data: TensorData,
}

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
    pub(crate) tensors: Vec<RecordTensor>,
    pub(crate) scalars: BTreeMap<String, Scalar>,
    pub(crate) paths: BTreeMap<String, String>,
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
        Ok(self.into_writer().into_bytes()?)
    }

    /// Reconstruct a record from an in-memory burnpack byte buffer.
    pub fn from_bytes(bytes: Bytes) -> Result<Self, RecordError> {
        Self::from_reader(Reader::from_bytes(bytes)?)
    }

    /// Save the record to a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn save<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), RecordError> {
        self.into_writer().write_to_file(path)?;
        Ok(())
    }

    /// Load a record from a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RecordError> {
        Self::from_reader(Reader::from_file(path)?)
    }

    fn into_writer(self) -> Writer {
        let tensors = self
            .tensors
            .into_iter()
            .map(|tensor| {
                burn_pack::Tensor::new(
                    tensor.name,
                    tensor.data.dtype,
                    tensor.data.shape,
                    tensor.param_id,
                    tensor.data.bytes,
                )
            })
            .collect();
        let mut writer = Writer::new(tensors);
        for (key, value) in &self.scalars {
            writer = writer.with_scalar(key, *value);
        }
        for (key, value) in &self.paths {
            writer = writer.with_metadata(key, value);
        }
        writer
    }

    fn from_reader(reader: Reader) -> Result<Self, RecordError> {
        let scalars = reader.scalars().clone();
        let paths = reader.metadata().clone();
        let tensors = reader
            .into_tensors()?
            .into_iter()
            .map(|tensor| {
                let (name, dtype, shape, param_id, bytes) = tensor.into_parts()?;
                Ok(RecordTensor {
                    name,
                    param_id,
                    data: TensorData::from_bytes(bytes, shape, dtype),
                })
            })
            .collect::<Result<Vec<_>, RecordError>>()?;
        Ok(Self {
            tensors,
            scalars,
            paths,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn record(data: TensorData) -> OptimizerRecord {
        OptimizerRecord {
            tensors: alloc::vec![RecordTensor {
                name: "42.moment".into(),
                param_id: Some(42),
                data,
            }],
            scalars: Default::default(),
            paths: Default::default(),
        }
    }

    fn assert_record(decoded: &OptimizerRecord, expected: &TensorData) {
        let tensor = &decoded.tensors[0];

        assert_eq!(tensor.name, "42.moment");
        assert_eq!(tensor.param_id, Some(42));
        assert_eq!(tensor.data.dtype, expected.dtype);
        assert_eq!(tensor.data.shape, expected.shape);
        assert_eq!(&tensor.data.bytes[..], &expected.bytes[..]);
    }

    #[test]
    fn tensor_data_stays_internal_across_burnpack_boundary() {
        let expected = TensorData::from([1.0_f32, 2.0, 3.0]);
        let decoded =
            OptimizerRecord::from_bytes(record(expected.clone()).into_bytes().unwrap()).unwrap();

        assert_record(&decoded, &expected);
    }

    #[cfg(feature = "std")]
    #[test]
    fn tensor_data_round_trips_through_burnpack_file() {
        let expected = TensorData::from([1.0_f32, 2.0, 3.0]);
        let path = std::env::temp_dir().join(format!(
            "burn_optim_record_{}_tensor_data.bpk",
            std::process::id()
        ));

        record(expected.clone()).save(&path).unwrap();
        let decoded = OptimizerRecord::load(&path).unwrap();
        assert_record(&decoded, &expected);

        drop(decoded);
        std::fs::remove_file(path).unwrap();
    }
}

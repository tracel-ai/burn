//! Non-generic record type for saving and loading module parameters.
//!
//! This is the new record system built on top of [`TensorSnapshot`] and the
//! burnpack format. Unlike the legacy `burn_core::record::Record` trait, which is
//! generic over a precision setting and materialized per-module by the derive
//! macro, [`RecordNew`] is a single concrete type. It holds lazy tensor snapshots
//! plus load-time configuration and serializes itself as a burnpack file.
//!
//! It is currently built *alongside* the legacy record system and is exposed via
//! [`crate::ModuleSnapshot::into_record_new`] / [`crate::ModuleSnapshot::load_record_new`].

use alloc::string::String;
use alloc::vec::Vec;

use crate::store::{PathFilter, TensorSnapshot};

/// Controls how a parameter's dtype is resolved when loading a [`RecordNew`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DTypePolicy {
    /// The module parameter adopts the record's dtype (data is loaded verbatim).
    ///
    /// This is the default and matches the behavior of the legacy applier: the
    /// loaded parameter takes whatever dtype was stored in the record. It also
    /// preserves lazy initialization on the load path.
    #[default]
    FromRecord,
    /// The record's data is cast to the module parameter's current dtype on load.
    ///
    /// Use this when the module's dtype is authoritative (e.g. a model already
    /// built at a specific precision) and the record's data should be converted
    /// to match. Note this materializes each target parameter to read its dtype.
    CastToModule,
}

/// Error returned by [`RecordNew`] save/load and apply operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordError {
    /// An I/O or format error occurred while reading or writing the record.
    Io(String),
    /// Validation failed while applying the record to a module (shape/dtype
    /// mismatch, or missing tensors when partial loading is not allowed).
    Validation(String),
}

impl core::fmt::Display for RecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RecordError::Io(msg) => write!(f, "Record I/O error: {msg}"),
            RecordError::Validation(msg) => write!(f, "Record validation error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RecordError {}

/// A non-generic record holding a module's parameters as lazy tensor snapshots.
///
/// Obtain one from a module with [`crate::ModuleSnapshot::into_record_new`], then
/// either save it ([`save`](RecordNew::save)/[`to_bytes`](RecordNew::to_bytes)) or
/// apply it back to a module with [`crate::ModuleSnapshot::load_record_new`].
///
/// Load-time behavior is configured with the builder methods
/// ([`with_dtype_policy`](RecordNew::with_dtype_policy),
/// [`with_filter`](RecordNew::with_filter),
/// [`allow_partial`](RecordNew::allow_partial),
/// [`validate`](RecordNew::validate)); these are ignored when saving.
///
/// The save-side dtype is intentionally not configurable: use `module.cast(dtype)`
/// before taking the record. The record stores whatever dtype the module holds.
#[derive(Clone)]
pub struct RecordNew {
    /// The lazily-materialized tensor snapshots that make up the record.
    pub(crate) snapshots: Vec<TensorSnapshot>,
    /// Policy controlling how dtypes are resolved on load. Default: [`DTypePolicy::FromRecord`].
    pub(crate) dtype_policy: DTypePolicy,
    /// Optional filter limiting which tensors are applied on load.
    pub(crate) filter: Option<PathFilter>,
    /// Allow loading even if some module parameters are missing from the record.
    pub(crate) allow_partial: bool,
    /// Validate shapes/dtypes while loading (errors abort the load).
    pub(crate) validate: bool,
}

impl core::fmt::Debug for RecordNew {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RecordNew")
            .field("num_tensors", &self.snapshots.len())
            .field("dtype_policy", &self.dtype_policy)
            .field("filter", &self.filter.is_some())
            .field("allow_partial", &self.allow_partial)
            .field("validate", &self.validate)
            .finish()
    }
}

impl RecordNew {
    /// Create a record from raw tensor snapshots with default load configuration.
    pub fn from_snapshots(snapshots: Vec<TensorSnapshot>) -> Self {
        Self {
            snapshots,
            dtype_policy: DTypePolicy::default(),
            filter: None,
            allow_partial: false,
            validate: true,
        }
    }

    /// The tensor snapshots held by this record.
    pub fn snapshots(&self) -> &[TensorSnapshot] {
        &self.snapshots
    }

    /// Consume the record and return its tensor snapshots.
    pub fn into_snapshots(self) -> Vec<TensorSnapshot> {
        self.snapshots
    }

    /// The number of tensors in the record.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether the record holds no tensors.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Set the dtype policy used when loading into a module.
    pub fn with_dtype_policy(mut self, policy: DTypePolicy) -> Self {
        self.dtype_policy = policy;
        self
    }

    /// Cast the record's data to the module parameter dtypes on load.
    ///
    /// Sugar for [`with_dtype_policy(DTypePolicy::CastToModule)`](RecordNew::with_dtype_policy).
    pub fn cast_to_module_dtype(self) -> Self {
        self.with_dtype_policy(DTypePolicy::CastToModule)
    }

    /// Restrict which tensors are applied on load using a [`PathFilter`].
    pub fn with_filter(mut self, filter: PathFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Allow loading even when some module parameters are absent from the record.
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }

    /// Enable or disable validation while loading.
    pub fn validate(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }
}

impl From<burn_pack::Error> for RecordError {
    fn from(err: burn_pack::Error) -> Self {
        use alloc::string::ToString;
        RecordError::Io(err.to_string())
    }
}

mod burnpack_io {
    use super::{RecordError, RecordNew};
    use crate::store::bridge;
    use alloc::vec::Vec;
    use burn_std::Bytes;
    use burn_pack::{Reader, Tensor as PackTensor, Writer};

    impl RecordNew {
        /// Serialize the record to an in-memory burnpack byte buffer.
        pub fn to_bytes(&self) -> Result<Bytes, RecordError> {
            let tensors: Vec<PackTensor> =
                self.snapshots.iter().map(bridge::snapshot_to_tensor).collect();
            let writer = Writer::new(tensors);
            Ok(writer.to_bytes()?)
        }

        /// Reconstruct a record from an in-memory burnpack byte buffer.
        ///
        /// Tensor data is materialized lazily; load-time configuration is reset
        /// to defaults.
        pub fn from_bytes(bytes: Bytes) -> Result<Self, RecordError> {
            let reader = Reader::from_bytes(bytes)?;
            let snapshots = reader
                .get_tensors()?
                .into_iter()
                .map(bridge::tensor_to_snapshot)
                .collect();
            Ok(Self::from_snapshots(snapshots))
        }

        /// Save the record to a burnpack file on disk.
        #[cfg(feature = "std")]
        pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), RecordError> {
            let tensors: Vec<PackTensor> =
                self.snapshots.iter().map(bridge::snapshot_to_tensor).collect();
            let writer = Writer::new(tensors);
            writer.write_to_file(path)?;
            Ok(())
        }

        /// Load a record from a burnpack file on disk.
        ///
        /// Tensor data is materialized lazily; load-time configuration is reset
        /// to defaults.
        #[cfg(feature = "std")]
        pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RecordError> {
            let reader = Reader::from_file(path)?;
            let snapshots = reader
                .get_tensors()?
                .into_iter()
                .map(bridge::tensor_to_snapshot)
                .collect();
            Ok(Self::from_snapshots(snapshots))
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::module::ParamId;
    use crate::tensor::{DType, TensorData};
    use alloc::vec;

    fn sample_snapshots() -> Vec<TensorSnapshot> {
        let weight = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]);
        let bias = TensorData::new(vec![5.0f32, 6.0], [2]);
        vec![
            TensorSnapshot::from_data(weight, vec!["weight".into()], vec![], ParamId::new()),
            TensorSnapshot::from_data(bias, vec!["bias".into()], vec![], ParamId::new()),
        ]
    }

    #[test]
    fn record_bytes_round_trip() {
        let record = RecordNew::from_snapshots(sample_snapshots());
        let bytes = record.to_bytes().expect("serialize to burnpack bytes");

        let loaded = RecordNew::from_bytes(bytes).expect("deserialize from burnpack bytes");
        assert_eq!(loaded.len(), 2);

        let mut by_name: Vec<_> = loaded
            .snapshots()
            .iter()
            .map(|s| (s.full_path(), s.clone()))
            .collect();
        by_name.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(by_name[0].0, "bias");
        assert_eq!(by_name[1].0, "weight");

        let weight = by_name[1].1.to_data().expect("materialize weight");
        assert_eq!(weight.shape.iter().copied().collect::<Vec<usize>>(), vec![2, 2]);
        assert_eq!(weight.dtype, DType::F32);
        assert_eq!(
            weight.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn record_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("record.bpk");

        let record = RecordNew::from_snapshots(sample_snapshots());
        record.save(&path).expect("save burnpack file");

        let loaded = RecordNew::load(&path).expect("load burnpack file");
        let bias = loaded
            .snapshots()
            .iter()
            .find(|s| s.full_path() == "bias")
            .expect("bias snapshot present")
            .to_data()
            .expect("materialize bias");
        assert_eq!(bias.to_vec::<f32>().unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn record_preserves_param_ids() {
        let snapshots = sample_snapshots();
        let original_ids: Vec<_> = snapshots.iter().map(|s| s.tensor_id).collect();

        let record = RecordNew::from_snapshots(snapshots);
        let bytes = record.to_bytes().unwrap();
        let loaded = RecordNew::from_bytes(bytes).unwrap();

        // Param ids survive the burnpack round-trip (important for stateful training).
        let mut loaded_sorted: Vec<_> = loaded.snapshots().to_vec();
        loaded_sorted.sort_by_key(|s| s.full_path());
        // sample_snapshots order is weight, bias; sorted is bias, weight
        assert_eq!(loaded_sorted[0].tensor_id, original_ids[1]);
        assert_eq!(loaded_sorted[1].tensor_id, original_ids[0]);
    }
}

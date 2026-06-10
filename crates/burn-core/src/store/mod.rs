//! Minimal, non-generic record system for saving and loading module parameters.
//!
//! A [`RecordNext`] holds a module's parameters (path + [`ParamId`] + [`TensorData`]) and
//! serializes them with the [burnpack](burn_pack) format. It is produced and applied with the
//! [`ModuleRecord`] extension trait ([`into_record_next`](ModuleRecord::into_record_next) /
//! [`load_record_next`](ModuleRecord::load_record_next)).
//!
//! This module is intentionally tiny: traversal is a straightforward [`ModuleVisitor`] /
//! [`ModuleMapper`] keyed by parameter path, with no filtering, adapters, or lazy snapshots.
//! The richer snapshot/import tooling (filtering, key remapping, PyTorch/SafeTensors adapters,
//! cross-framework stores) lives in the `burn-store` crate.
//!
//! It is built *alongside* the legacy [`crate::record`] system.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use crate::tensor::{Bool, DType, Device, Int, Shape, Tensor, TensorData, kind::Basic};

use burn_pack::{Reader, Writer};

/// Controls how a parameter's dtype is resolved when loading a [`RecordNext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DTypePolicy {
    /// The module parameter adopts the record's dtype (data is loaded verbatim). Default.
    #[default]
    FromRecord,
    /// The record's data is cast to the module parameter's current dtype on load.
    ///
    /// Note this materializes each target parameter to read its dtype.
    CastToModule,
}

/// Error returned by [`RecordNext`] save/load and [`ModuleRecord`] apply operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordError {
    /// An I/O or format error occurred while reading or writing the record.
    Io(String),
    /// Validation failed while applying the record (shape mismatch, or missing tensors
    /// when partial loading is not allowed).
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

impl From<burn_pack::Error> for RecordError {
    fn from(err: burn_pack::Error) -> Self {
        RecordError::Io(err.to_string())
    }
}

/// A single recorded tensor: its module path, parameter id, and data.
#[derive(Clone)]
struct RecordTensor {
    path: String,
    id: ParamId,
    data: TensorData,
}

/// A non-generic record holding a module's parameters.
///
/// Obtain one from a module with [`ModuleRecord::into_record_next`], then either save it
/// ([`save`](RecordNext::save) / [`to_bytes`](RecordNext::to_bytes)) or apply it back with
/// [`ModuleRecord::load_record_next`]. Load-time behavior is configured with the builder
/// methods; they are ignored when saving.
///
/// The save-side dtype is intentionally not configurable: use `module.cast(dtype)` before
/// taking the record. The record stores whatever dtype the module currently holds.
#[derive(Clone)]
pub struct RecordNext {
    tensors: Vec<RecordTensor>,
    dtype_policy: DTypePolicy,
    allow_partial: bool,
    validate: bool,
}

impl core::fmt::Debug for RecordNext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RecordNext")
            .field("num_tensors", &self.tensors.len())
            .field("dtype_policy", &self.dtype_policy)
            .field("allow_partial", &self.allow_partial)
            .field("validate", &self.validate)
            .finish()
    }
}

impl RecordNext {
    fn from_tensors(tensors: Vec<RecordTensor>) -> Self {
        Self {
            tensors,
            dtype_policy: DTypePolicy::default(),
            allow_partial: false,
            validate: true,
        }
    }

    /// The number of tensors in the record.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the record holds no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Set the dtype policy used when loading into a module.
    pub fn with_dtype_policy(mut self, policy: DTypePolicy) -> Self {
        self.dtype_policy = policy;
        self
    }

    /// Cast the record's data to the module parameter dtypes on load.
    ///
    /// Sugar for [`with_dtype_policy(DTypePolicy::CastToModule)`](RecordNext::with_dtype_policy).
    pub fn cast_to_module_dtype(self) -> Self {
        self.with_dtype_policy(DTypePolicy::CastToModule)
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

    /// Serialize the record to an in-memory burnpack byte buffer.
    pub fn to_bytes(&self) -> Result<crate::tensor::Bytes, RecordError> {
        Ok(Writer::new(self.pack_tensors()).to_bytes()?)
    }

    /// Reconstruct a record from an in-memory burnpack byte buffer.
    pub fn from_bytes(bytes: crate::tensor::Bytes) -> Result<Self, RecordError> {
        Self::from_reader(Reader::from_bytes(bytes)?)
    }

    /// Save the record to a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), RecordError> {
        Writer::new(self.pack_tensors()).write_to_file(path)?;
        Ok(())
    }

    /// Load a record from a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RecordError> {
        Self::from_reader(Reader::from_file(path)?)
    }

    fn pack_tensors(&self) -> Vec<burn_pack::Tensor> {
        self.tensors
            .iter()
            .map(|t| {
                burn_pack::Tensor::new(
                    t.path.clone(),
                    t.data.dtype,
                    t.data.shape.clone(),
                    Some(t.id.val()),
                    t.data.bytes.clone(),
                )
            })
            .collect()
    }

    fn from_reader(reader: Reader) -> Result<Self, RecordError> {
        let tensors = reader
            .get_tensors()?
            .into_iter()
            .map(|t| {
                let id = t.param_id.map(ParamId::from).unwrap_or_else(ParamId::new);
                let data = TensorData::from_bytes(t.bytes, t.shape, t.dtype);
                RecordTensor {
                    path: t.name,
                    id,
                    data,
                }
            })
            .collect();
        Ok(Self::from_tensors(tensors))
    }
}

/// Extension trait adding the non-generic record API to every [`Module`].
pub trait ModuleRecord: Module {
    /// Collect this module's parameters into a [`RecordNext`].
    fn into_record_next(&self) -> RecordNext {
        let mut collector = Collector::default();
        self.visit(&mut collector);
        RecordNext::from_tensors(collector.tensors)
    }

    /// Apply a [`RecordNext`] to this module, returning the loaded module.
    ///
    /// Honors the record's [`DTypePolicy`], `validate`, and `allow_partial` settings.
    fn apply_record(self, record: RecordNext) -> Result<Self, RecordError>
    where
        Self: Sized,
    {
        let validate = record.validate;
        let allow_partial = record.allow_partial;

        let mut applier = Applier::new(record);
        let module = self.map(&mut applier);

        if validate && !applier.errors.is_empty() {
            return Err(RecordError::Validation(format!(
                "Apply errors: {:?}",
                applier.errors
            )));
        }
        if !allow_partial && !applier.missing.is_empty() {
            return Err(RecordError::Validation(format!(
                "Missing tensors: {:?}",
                applier.missing
            )));
        }

        Ok(module)
    }

    /// Apply a [`RecordNext`] to this module, consuming and returning it.
    ///
    /// Panics if validation fails; use [`apply_record`](Self::apply_record) for the fallible
    /// variant.
    fn load_record_next(self, record: RecordNext) -> Self
    where
        Self: Sized,
    {
        self.apply_record(record).expect("Failed to load record")
    }
}

impl<M: Module> ModuleRecord for M {}

/// Visitor that collects every parameter as a [`RecordTensor`], keyed by its module path.
#[derive(Default)]
struct Collector {
    path: Vec<String>,
    tensors: Vec<RecordTensor>,
}

impl Collector {
    fn record(&mut self, id: ParamId, data: TensorData) {
        self.tensors.push(RecordTensor {
            path: self.path.join("."),
            id,
            data,
        });
    }
}

impl ModuleVisitor for Collector {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.record(param.id, param.val().into_data());
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        self.record(param.id, param.val().into_data());
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
        self.record(param.id, param.val().into_data());
    }
}

/// Mapper that loads recorded tensors back onto matching parameters by module path.
struct Applier {
    path: Vec<String>,
    tensors: HashMap<String, TensorData>,
    dtype_policy: DTypePolicy,
    missing: Vec<String>,
    errors: Vec<String>,
}

impl Applier {
    fn new(record: RecordNext) -> Self {
        let tensors = record
            .tensors
            .into_iter()
            .map(|t| (t.path, t.data))
            .collect();
        Self {
            path: Vec::new(),
            tensors,
            dtype_policy: record.dtype_policy,
            missing: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Look up the recorded tensor for the current path and build the tensor to load,
    /// or `None` (recording it as missing / errored) to leave the parameter unchanged.
    fn take<const D: usize, K: Basic>(
        &mut self,
        device: &Device,
        target_shape: Shape,
        target_dtype: Option<DType>,
    ) -> Option<Tensor<D, K>> {
        let path = self.path.join(".");
        let mut data = match self.tensors.get(&path) {
            Some(data) => data.clone(),
            None => {
                self.missing.push(path);
                return None;
            }
        };

        // Resolve the dtype to load with (CastToModule casts to the module's dtype).
        let dtype = target_dtype.unwrap_or(data.dtype);
        if data.dtype != dtype {
            data = data.convert_dtype(dtype);
        }

        if data.shape != target_shape {
            self.errors.push(format!(
                "{path}: shape mismatch, expected {:?} but record has {:?}",
                target_shape, data.shape
            ));
            return None;
        }

        Some(Tensor::from_data(data, (device, dtype)))
    }
}

impl ModuleMapper for Applier {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let id = param.id;
        let device = param.lazy_device();
        let shape = param.lazy_shape();
        let target_dtype = match self.dtype_policy {
            DTypePolicy::FromRecord => None,
            DTypePolicy::CastToModule => Some(param.val().dtype()),
        };
        match self.take(&device, shape, target_dtype) {
            Some(tensor) => param.transform_for_load(tensor, id),
            None => param,
        }
    }

    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        let id = param.id;
        let device = param.lazy_device();
        let shape = param.lazy_shape();
        let target_dtype = match self.dtype_policy {
            DTypePolicy::FromRecord => None,
            DTypePolicy::CastToModule => Some(param.val().dtype()),
        };
        match self.take(&device, shape, target_dtype) {
            Some(tensor) => param.transform_for_load(tensor, id),
            None => param,
        }
    }

    fn map_bool<const D: usize>(&mut self, param: Param<Tensor<D, Bool>>) -> Param<Tensor<D, Bool>> {
        let id = param.id;
        let device = param.lazy_device();
        let shape = param.lazy_shape();
        let target_dtype = match self.dtype_policy {
            DTypePolicy::FromRecord => None,
            DTypePolicy::CastToModule => Some(param.val().dtype()),
        };
        match self.take(&device, shape, target_dtype) {
            Some(tensor) => param.transform_for_load(tensor, id),
            None => param,
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate as burn;
    use crate::module::{Module, Param};
    use crate::tensor::Tensor;
    use burn_tensor::Device;

    #[derive(Module, Debug)]
    struct Tiny {
        weight: Param<Tensor<2>>,
        bias: Param<Tensor<1>>,
    }

    impl Tiny {
        fn new(weight: [[f32; 2]; 2], bias: [f32; 2], device: &Device) -> Self {
            Self {
                weight: Param::from_data(weight, device),
                bias: Param::from_data(bias, device),
            }
        }
    }

    #[derive(Module, Debug)]
    struct TinyWide {
        weight: Param<Tensor<2>>,
        bias: Param<Tensor<1>>,
        gamma: Param<Tensor<1>>,
    }

    impl TinyWide {
        fn zeros(device: &Device) -> Self {
            Self {
                weight: Param::from_data([[0.0, 0.0], [0.0, 0.0]], device),
                bias: Param::from_data([0.0, 0.0], device),
                gamma: Param::from_data([0.0, 0.0], device),
            }
        }
    }

    fn weights(model: &Tiny) -> (Vec<f32>, Vec<f32>) {
        (
            model.weight.val().to_data().to_vec().unwrap(),
            model.bias.val().to_data().to_vec().unwrap(),
        )
    }

    #[test]
    fn round_trip_in_memory() {
        let device = Default::default();
        let model = Tiny::new([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0], &device);

        let bytes = model.into_record_next().to_bytes().unwrap();
        let record = RecordNext::from_bytes(bytes).unwrap();
        assert_eq!(record.len(), 2);

        let loaded = Tiny::new([[0.0; 2]; 2], [0.0; 2], &device).load_record_next(record);
        let (w, b) = weights(&loaded);
        assert_eq!(w, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b, vec![5.0, 6.0]);
    }

    #[test]
    fn round_trip_file() {
        let device = Default::default();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.bpk");

        Tiny::new([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0], &device)
            .into_record_next()
            .save(&path)
            .unwrap();

        let record = RecordNext::load(&path).unwrap();
        let loaded = Tiny::new([[0.0; 2]; 2], [0.0; 2], &device).load_record_next(record);
        let (w, b) = weights(&loaded);
        assert_eq!(w, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b, vec![5.0, 6.0]);
    }

    #[test]
    fn missing_tensor_requires_allow_partial() {
        let device = Default::default();
        // Tiny has weight+bias; TinyWide also expects `gamma`, which the record lacks.
        let record = Tiny::new([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0], &device).into_record_next();

        let strict = TinyWide::zeros(&device).apply_record(record.clone());
        assert!(matches!(strict, Err(RecordError::Validation(_))));

        let partial = TinyWide::zeros(&device).apply_record(record.allow_partial(true));
        assert!(partial.is_ok());
        let loaded = partial.unwrap();
        // weight/bias were loaded; gamma kept its (zero) initialization.
        assert_eq!(
            loaded.weight.val().to_data().to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            loaded.gamma.val().to_data().to_vec::<f32>().unwrap(),
            vec![0.0, 0.0]
        );
    }
}

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_core as burn;

use burn::config::Config;
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::store::{DTypePolicy, ModuleRecord, RecordError};
use burn::tensor::{
    Bool, Bytes, DType, Device, FloatDType, Int, Tensor, container::TensorContainer,
};
use burn_pack::{Reader, Scalar, Writer};
use hashbrown::HashMap;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Scalar key under which the EMA update step counter is persisted in a burnpack container.
const STEP_KEY: &str = "step";

/// Decay warmup schedule for [`ExponentialMovingAverage`].
///
/// Without warmup, a high target decay (such as `0.999` or `0.9999`) anchors the shadow parameters
/// to the random initial weights for thousands of steps. Warming the decay up from a small value
/// lets early updates replace the initialization quickly before settling at the target decay.
#[derive(Config, Debug, Copy, PartialEq)]
pub enum EmaWarmup {
    /// Use the configured target `decay` unmodified from the very first update.
    None,

    /// Rational warmup `min(decay, (1 + step) / (warmup + step))` where `step` is the 0-indexed
    /// update count (`0` on the first update).
    ///
    /// A `warmup` of `10.0` (the default, used by `timm` and TensorFlow) gives an effective decay
    /// of `1 / 10 = 0.1` on the first update, `2 / 11` on the second, and approaches `1.0`
    /// asymptotically until capped by `decay`.
    Linear {
        /// Denominator offset (`>= 1.0`). Larger values warm up more slowly.
        warmup: f64,
    },

    /// Exact in-place exponential bias correction `1 - (1 - decay) / (1 - decay^(step + 1))`
    /// where `step` is the 0-indexed update count (`0` on the first update).
    ///
    /// On the first update (`step = 0`), the effective decay is `0.0`, so the shadow parameters
    /// copy the post-step model weights directly with zero weight on the random initialization.
    /// Subsequent steps maintain the exact bias-corrected moving average without allocating a
    /// separate uncorrected buffer.
    BiasCorrection,
}

impl Default for EmaWarmup {
    fn default() -> Self {
        Self::Linear { warmup: 10.0 }
    }
}

impl EmaWarmup {
    /// Compute the effective decay rate for the given target `decay` and 0-indexed `step`.
    pub fn effective_decay(&self, decay: f64, step: usize) -> f64 {
        match *self {
            Self::None => decay,
            Self::Linear { warmup } => {
                let s = step as f64;
                decay.min((1.0 + s) / (warmup + s))
            }
            Self::BiasCorrection => {
                if decay <= 0.0 {
                    return 0.0;
                }
                if decay >= 1.0 {
                    return 1.0;
                }
                let t = (step + 1) as f64;
                let denom = 1.0 - decay.powf(t);
                if denom <= 0.0 {
                    decay
                } else {
                    (1.0 - (1.0 - decay) / denom).clamp(0.0, decay)
                }
            }
        }
    }

    fn validate(&self) {
        if let Self::Linear { warmup } = *self {
            assert!(
                warmup.is_finite() && warmup >= 1.0,
                "EmaWarmup::Linear warmup must be finite and >= 1.0, got {warmup}."
            );
        }
    }
}

/// Configuration to create an [`ExponentialMovingAverage`].
#[derive(Config, Debug)]
pub struct ExponentialMovingAverageConfig {
    /// Target exponential decay rate in `[0.0, 1.0]`.
    #[config(default = 0.999)]
    pub decay: f64,

    /// Warmup schedule applied to `decay` on early updates.
    #[config(default = "EmaWarmup::default()")]
    pub warmup: EmaWarmup,
}

/// Type alias for [`ExponentialMovingAverageConfig`].
pub type EmaConfig = ExponentialMovingAverageConfig;

impl ExponentialMovingAverageConfig {
    /// Initialize an [`ExponentialMovingAverage`] shadow tracker from `model`.
    ///
    /// The shadow module is initialized via [`Module::valid`], so its parameters are detached
    /// from the autodiff graph and its training flags are disabled for inference.
    ///
    /// # Panics
    ///
    /// Panics if `decay` is not finite or outside `[0.0, 1.0]`, or if `warmup` is
    /// [`EmaWarmup::Linear`] with a value less than `1.0`.
    pub fn init<M: Module>(&self, model: &M) -> ExponentialMovingAverage<M> {
        assert!(
            self.decay.is_finite() && (0.0..=1.0).contains(&self.decay),
            "EMA decay must be finite and within [0.0, 1.0], got {}.",
            self.decay
        );
        self.warmup.validate();

        ExponentialMovingAverage {
            shadow: model.valid(),
            decay: self.decay,
            warmup: self.warmup,
            step: 0,
        }
    }
}

/// Exponential moving average (EMA) of a module's floating-point parameters:
///
/// ```text
/// shadow <- effective_decay * shadow + (1 - effective_decay) * param
/// ```
///
/// Call [`update`](Self::update) after each optimizer step, [`model`](Self::model) to obtain a
/// detached validation module for evaluation or inference, and [`to_record`](Self::to_record) /
/// [`load_record`](Self::load_record) to checkpoint and resume training.
///
/// # Details
///
/// - **Detached shadow:** The shadow module is kept on the non-autodiff inner backend with
///   `require_grad = false`, avoiding autodiff dispatch overhead and graph retention.
/// - **Device and dtype tracking:** If `model` moves to another device or changes floating-point
///   dtype between updates, the shadow parameters follow automatically. Half-precision (`F16` and
///   `BF16`) updates compute the linear interpolation in `F32` before casting back so high decay
///   rates (`0.999`, `0.9999`) do not stagnate from mantissa underflow.
/// - **Decay warmup:** Configured through [`EmaWarmup`] (defaulting to [`EmaWarmup::Linear`] with
///   `warmup = 10.0`) so the shadow is not anchored to random initialization.
#[derive(Clone, Debug)]
pub struct ExponentialMovingAverage<M: Module> {
    shadow: M,
    decay: f64,
    warmup: EmaWarmup,
    step: usize,
}

/// Type alias for [`ExponentialMovingAverage`].
pub type Ema<M> = ExponentialMovingAverage<M>;

impl<M: Module> ExponentialMovingAverage<M> {
    /// Update the shadow parameters from `model`, advancing the step counter by one.
    pub fn update(&mut self, model: &M) {
        let decay = self.effective_decay();

        let mut collector = EmaCollector::default();
        model.visit(&mut collector);

        let mut mapper = EmaMapper {
            path: Vec::new(),
            decay,
            by_id: collector.by_id,
            by_path: collector.by_path,
        };
        self.shadow = self.shadow.clone().map(&mut mapper);
        self.step += 1;
    }

    /// Return an isolated validation snapshot of the averaged module.
    ///
    /// The returned module is detached from autodiff and has training flags disabled, so it can
    /// be passed directly to evaluation or saved with [`Module::into_record`].
    pub fn model(&self) -> M {
        self.shadow.valid()
    }

    /// Consume the EMA tracker and return the averaged module.
    pub fn into_model(self) -> M {
        self.shadow
    }

    /// The configured target decay rate.
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// The configured decay warmup schedule.
    pub fn warmup(&self) -> EmaWarmup {
        self.warmup
    }

    /// The number of [`update`](Self::update) steps performed so far.
    pub fn step(&self) -> usize {
        self.step
    }

    /// The effective decay rate that will be used on the next [`update`](Self::update) call.
    pub fn effective_decay(&self) -> f64 {
        self.warmup.effective_decay(self.decay, self.step)
    }

    /// Move the shadow module to `device` (without autodiff).
    pub fn to_device(mut self, device: &Device) -> Self {
        let plain_device = device.clone().without_autodiff();
        self.shadow = self.shadow.to_device(&plain_device);
        self
    }

    /// Collect the shadow module and step counter into a serializable [`EmaRecord`].
    pub fn to_record(&self) -> EmaRecord {
        EmaRecord {
            model: self.shadow.valid().into_record(),
            step: self.step,
        }
    }

    /// Apply an [`EmaRecord`] to this EMA tracker, returning an error if record validation fails.
    pub fn try_load_record(mut self, record: EmaRecord) -> Result<Self, RecordError> {
        self.shadow = self.shadow.try_load_record(record.model)?.valid();
        self.step = record.step;
        Ok(self)
    }

    /// Apply an [`EmaRecord`] to this EMA tracker.
    ///
    /// Panics if record validation fails; use [`try_load_record`](Self::try_load_record) for the
    /// fallible variant.
    pub fn load_record(self, record: EmaRecord) -> Self {
        self.try_load_record(record)
            .expect("Failed to load EMA record")
    }

    /// Serialize the EMA state to an in-memory burnpack byte buffer.
    pub fn into_bytes(&self) -> Result<Bytes, RecordError> {
        self.to_record().into_bytes()
    }

    /// Restore the EMA state from an in-memory burnpack byte buffer.
    pub fn from_bytes(self, bytes: Bytes) -> Result<Self, RecordError> {
        let record = EmaRecord::from_bytes(bytes)?;
        self.try_load_record(record)
    }

    /// Save the EMA state to a burnpack file on disk.
    ///
    /// Because [`EmaRecord`] stores the shadow parameters under their standard module paths and
    /// records `step` as a burnpack scalar, the saved file can also be loaded directly into a
    /// standalone module via [`Module::load_file`].
    #[cfg(feature = "std")]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), RecordError> {
        self.to_record().save(path)
    }

    /// Load the EMA state from a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn load<P: AsRef<std::path::Path>>(self, path: P) -> Result<Self, RecordError> {
        let record = EmaRecord::load(path)?;
        self.try_load_record(record)
    }
}

/// Serialized state of an [`ExponentialMovingAverage`], stored in the [burnpack](burn_pack)
/// format.
///
/// Holds the shadow [`ModuleRecord`] together with the update `step` counter. When written to a
/// burnpack buffer or file, the shadow tensors are stored under their module paths alongside a
/// `"step"` scalar entry, making the container loadable both by [`EmaRecord`] and directly by
/// [`ModuleRecord`].
#[derive(Clone, Debug)]
pub struct EmaRecord {
    model: ModuleRecord,
    step: usize,
}

impl EmaRecord {
    /// Create an [`EmaRecord`] from a shadow [`ModuleRecord`] and update `step` count.
    pub fn new(model: ModuleRecord, step: usize) -> Self {
        Self { model, step }
    }

    /// Borrow the underlying shadow [`ModuleRecord`].
    pub fn model(&self) -> &ModuleRecord {
        &self.model
    }

    /// Consume the record and return the underlying shadow [`ModuleRecord`].
    pub fn into_model(self) -> ModuleRecord {
        self.model
    }

    /// The number of EMA updates recorded in this checkpoint.
    pub fn step(&self) -> usize {
        self.step
    }

    /// The number of tensors in the shadow module record.
    pub fn len(&self) -> usize {
        self.model.len()
    }

    /// Whether the shadow module record holds no tensors.
    pub fn is_empty(&self) -> bool {
        self.model.is_empty()
    }

    /// Set the [`DTypePolicy`] used when applying the shadow module record.
    pub fn with_dtype_policy(mut self, policy: DTypePolicy) -> Self {
        self.model = self.model.with_dtype_policy(policy);
        self
    }

    /// Allow loading even when some shadow module parameters are absent from the record.
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.model = self.model.allow_partial(allow);
        self
    }

    /// Allow record tensors that do not match any shadow module parameter.
    pub fn allow_unused(mut self, allow: bool) -> Self {
        self.model = self.model.allow_unused(allow);
        self
    }

    /// Enable or disable shape validation when applying the shadow module record.
    pub fn validate(mut self, validate: bool) -> Self {
        self.model = self.model.validate(validate);
        self
    }

    /// Serialize the record to an in-memory burnpack byte buffer.
    pub fn into_bytes(self) -> Result<Bytes, RecordError> {
        Ok(self.into_writer()?.into_bytes()?)
    }

    /// Reconstruct a record from an in-memory burnpack byte buffer.
    pub fn from_bytes(bytes: Bytes) -> Result<Self, RecordError> {
        let reader = Reader::from_bytes(bytes.clone())?;
        let step = read_step(&reader);
        let model = ModuleRecord::from_bytes(bytes)?;
        Ok(Self { model, step })
    }

    /// Save the record to a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn save<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), RecordError> {
        self.into_writer()?.write_to_file(path)?;
        Ok(())
    }

    /// Load a record from a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RecordError> {
        let reader = Reader::from_file(&path)?;
        let step = read_step(&reader);
        let model = ModuleRecord::load(path)?;
        Ok(Self { model, step })
    }

    fn into_writer(self) -> Result<Writer, RecordError> {
        let model_bytes = self.model.into_bytes()?;
        let tensors = Reader::from_bytes(model_bytes)?.into_tensors()?;
        Ok(Writer::new(tensors).with_scalar(STEP_KEY, Scalar::from(self.step)))
    }
}

impl From<ModuleRecord> for EmaRecord {
    fn from(model: ModuleRecord) -> Self {
        Self { model, step: 0 }
    }
}

fn read_step(reader: &Reader) -> usize {
    reader
        .scalars()
        .get(STEP_KEY)
        .copied()
        .and_then(|scalar| usize::try_from(scalar).ok())
        .unwrap_or(0)
}

#[derive(Default)]
struct EmaCollector {
    path: Vec<String>,
    by_id: TensorContainer<ParamId>,
    by_path: HashMap<String, ParamId>,
}

impl EmaCollector {
    fn record_path(&mut self, id: ParamId) {
        self.by_path.insert(self.path.join("."), id);
    }
}

impl ModuleVisitor for EmaCollector {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let tensor = param.val().without_autodiff().set_require_grad(false);
        self.record_path(param.id);
        self.by_id.register(param.id, tensor);
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        let tensor = param.val().without_autodiff();
        self.record_path(param.id);
        self.by_id.register(param.id, tensor);
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
        let tensor = param.val().without_autodiff();
        self.record_path(param.id);
        self.by_id.register(param.id, tensor);
    }
}

struct EmaMapper {
    path: Vec<String>,
    decay: f64,
    by_id: TensorContainer<ParamId>,
    by_path: HashMap<String, ParamId>,
}

impl EmaMapper {
    fn take_model_tensor<T: Clone + Send + 'static>(
        &mut self,
        id: ParamId,
    ) -> Option<(ParamId, T)> {
        if let Some(tensor) = self.by_id.remove::<T>(&id) {
            return Some((id, tensor));
        }
        let path = self.path.join(".");
        let model_id = self.by_path.remove(&path)?;
        let tensor = self.by_id.remove::<T>(&model_id)?;
        Some((model_id, tensor))
    }
}

impl ModuleMapper for EmaMapper {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let Some((resolved_id, model_tensor)) = self.take_model_tensor::<Tensor<D>>(param.id)
        else {
            return param;
        };

        let (_, shadow_tensor, param_mapper) = param.consume();
        let target_device = model_tensor.device();
        let target_dtype = model_tensor.dtype();

        if matches!(target_dtype, DType::QFloat(_)) {
            return Param::from_mapped_value(
                resolved_id,
                model_tensor.without_autodiff().set_require_grad(false),
                param_mapper,
            );
        }

        let shadow_tensor = if shadow_tensor.device() != target_device {
            shadow_tensor.to_device(&target_device)
        } else {
            shadow_tensor
        };

        let decay = self.decay;
        let one_minus_decay = 1.0 - decay;

        let updated = match target_dtype {
            DType::F16 | DType::BF16 => {
                let shadow_f32 = shadow_tensor.cast(FloatDType::F32);
                let model_f32 = model_tensor.cast(FloatDType::F32);
                shadow_f32
                    .mul_scalar(decay)
                    .add(model_f32.mul_scalar(one_minus_decay))
                    .cast(target_dtype)
            }
            _ => {
                let shadow_cast = if shadow_tensor.dtype() != target_dtype {
                    shadow_tensor.cast(target_dtype)
                } else {
                    shadow_tensor
                };
                shadow_cast
                    .mul_scalar(decay)
                    .add(model_tensor.mul_scalar(one_minus_decay))
            }
        };

        Param::from_mapped_value(
            resolved_id,
            updated.without_autodiff().set_require_grad(false),
            param_mapper,
        )
    }

    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        let Some((resolved_id, model_tensor)) = self.take_model_tensor::<Tensor<D, Int>>(param.id)
        else {
            return param;
        };
        let (_, _, param_mapper) = param.consume();
        Param::from_mapped_value(resolved_id, model_tensor.without_autodiff(), param_mapper)
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<D, Bool>>,
    ) -> Param<Tensor<D, Bool>> {
        let Some((resolved_id, model_tensor)) = self.take_model_tensor::<Tensor<D, Bool>>(param.id)
        else {
            return param;
        };
        let (_, _, param_mapper) = param.consume();
        Param::from_mapped_value(resolved_id, model_tensor.without_autodiff(), param_mapper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GradientsParams, SgdConfig};
    use burn::module::RunningState;
    use burn::tensor::{TensorData, Tolerance};
    use burn_derive::Module;
    use burn_nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};

    fn make_linear(weights: [[f32; 2]; 2], bias: [f32; 2], device: &Device) -> Linear {
        let record = Linear {
            weight: Param::from_data(weights, device),
            bias: Some(Param::from_data(bias, device)),
        }
        .into_record();
        LinearConfig::new(2, 2).init(device).load_record(record)
    }

    #[test]
    fn constant_decay_tracks_exponential_moving_average() {
        let device = Device::default().autodiff();
        let mut model = make_linear([[1.0, 2.0], [3.0, 4.0]], [0.5, -0.5], &device);

        let mut ema = ExponentialMovingAverageConfig::new()
            .with_decay(0.8)
            .with_warmup(EmaWarmup::None)
            .init(&model);

        assert_eq!(ema.step(), 0);
        assert!(!ema.model().weight.val().device().is_autodiff());
        assert!(!ema.model().weight.is_require_grad());

        // Step 1: update model weights to [[10.0, 20.0], [30.0, 40.0]], bias [5.0, -5.0]
        model.weight = Param::initialized(
            model.weight.id,
            Tensor::<2>::from_floats([[10.0, 20.0], [30.0, 40.0]], &device).require_grad(),
        );
        model.bias = Some(Param::initialized(
            model.bias.as_ref().unwrap().id,
            Tensor::<1>::from_floats([5.0, -5.0], &device).require_grad(),
        ));
        ema.update(&model);

        assert_eq!(ema.step(), 1);
        // 0.8 * [1, 2, 3, 4] + 0.2 * [10, 20, 30, 40] = [2.8, 5.6, 8.4, 11.2]
        let shadow = ema.model();
        shadow.weight.val().into_data().assert_approx_eq::<f32>(
            &TensorData::from([[2.8_f32, 5.6], [8.4, 11.2]]),
            Tolerance::absolute(1e-5),
        );
        shadow
            .bias
            .unwrap()
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([1.4_f32, -1.4]),
                Tolerance::absolute(1e-5),
            );
        assert!(!shadow.weight.val().device().is_autodiff());
        assert!(!shadow.weight.is_require_grad());
    }

    #[test]
    fn linear_warmup_avoids_anchoring_to_initialization() {
        let device = Device::default().autodiff();
        let mut model = make_linear([[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0], &device);

        // Target decay is 0.999, but default linear warmup (10.0) gives effective decay 0.1 on step 0.
        let mut ema = ExponentialMovingAverageConfig::new()
            .with_decay(0.999)
            .init(&model);

        assert!((ema.effective_decay() - 0.1).abs() < 1e-12);

        model.weight = Param::initialized(
            model.weight.id,
            Tensor::<2>::from_floats([[10.0, 10.0], [10.0, 10.0]], &device).require_grad(),
        );
        ema.update(&model);

        // With effective decay 0.1, shadow becomes 0.1 * 0 + 0.9 * 10 = 9.0 (not 0.01).
        ema.model()
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[9.0_f32, 9.0], [9.0, 9.0]]),
                Tolerance::absolute(1e-5),
            );
        // Step 1 effective decay: (1 + 1) / (10 + 1) = 2 / 11
        assert!((ema.effective_decay() - (2.0 / 11.0)).abs() < 1e-12);
    }

    #[test]
    fn bias_correction_warmup_matches_exact_debiased_ema() {
        let device = Device::default();
        let decay = 0.9_f64;
        let model = make_linear([[100.0, 100.0], [100.0, 100.0]], [0.0, 0.0], &device);

        let mut ema = ExponentialMovingAverageConfig::new()
            .with_decay(decay)
            .with_warmup(EmaWarmup::BiasCorrection)
            .init(&model);

        let values = [1.0_f32, 2.0, 4.0, 8.0];
        let mut raw_moment = 0.0_f64;
        for (i, &v) in values.iter().enumerate() {
            let step_model = make_linear([[v, v], [v, v]], [v, v], &device);
            ema.update(&step_model);

            raw_moment = decay * raw_moment + (1.0 - decay) * (v as f64);
            let expected = (raw_moment / (1.0 - decay.powi((i + 1) as i32))) as f32;

            ema.model()
                .weight
                .val()
                .into_data()
                .assert_approx_eq::<f32>(
                    &TensorData::from([[expected, expected], [expected, expected]]),
                    Tolerance::absolute(1e-5),
                );
        }
    }

    #[test]
    fn half_precision_updates_do_not_underflow_at_high_decay() {
        let device = Device::default();
        let mut model = Linear {
            weight: Param::from_tensor(
                Tensor::<2>::from_floats([[1.0, 1.0], [1.0, 1.0]], &device).cast(FloatDType::BF16),
            ),
            bias: None,
        };

        let mut ema = ExponentialMovingAverageConfig::new()
            .with_decay(0.99)
            .with_warmup(EmaWarmup::None)
            .init(&model);

        model.weight = Param::initialized(
            model.weight.id,
            Tensor::<2>::from_floats([[101.0, 101.0], [101.0, 101.0]], &device)
                .cast(FloatDType::BF16),
        );
        ema.update(&model);

        let shadow_weight = ema.model().weight.val();
        assert_eq!(shadow_weight.dtype(), DType::BF16);
        // 0.99 * 1.0 + 0.01 * 101.0 = 2.0 (in pure BF16, 0.99 * 1.0 rounds to 0.98828125).
        shadow_weight
            .cast(FloatDType::F32)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[2.0_f32, 2.0], [2.0, 2.0]]),
                Tolerance::absolute(1e-2),
            );
    }

    #[test]
    fn follows_model_dtype_change_and_matches_fresh_model_by_path() {
        let device = Device::default();
        let model_f32 = make_linear([[2.0, 4.0], [6.0, 8.0]], [1.0, 2.0], &device);

        let mut ema = ExponentialMovingAverageConfig::new()
            .with_decay(0.5)
            .with_warmup(EmaWarmup::None)
            .init(&model_f32);

        // Construct a brand-new Linear with different ParamIds and F16 weights.
        let model_f16 = Linear {
            weight: Param::from_tensor(
                Tensor::<2>::from_floats([[6.0, 8.0], [10.0, 12.0]], &device).cast(FloatDType::F16),
            ),
            bias: Some(Param::from_tensor(
                Tensor::<1>::from_floats([3.0, 4.0], &device).cast(FloatDType::F16),
            )),
        };

        ema.update(&model_f16);

        let shadow = ema.model();
        assert_eq!(shadow.weight.val().dtype(), DType::F16);
        assert_eq!(shadow.weight.id, model_f16.weight.id);
        shadow
            .weight
            .val()
            .cast(FloatDType::F32)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[4.0_f32, 6.0], [8.0, 10.0]]),
                Tolerance::absolute(1e-2),
            );
    }

    #[test]
    fn snapshot_isolation_with_running_state() {
        #[derive(Module, Debug)]
        struct WithRunning {
            bn: BatchNorm,
            counter: RunningState<Tensor<1>>,
        }

        let device = Device::default();
        let module = WithRunning {
            bn: BatchNormConfig::new(2).init(&device),
            counter: RunningState::new(Tensor::from_floats([1.0, 2.0], &device)),
        };

        let mut ema = ExponentialMovingAverageConfig::new()
            .with_decay(0.5)
            .with_warmup(EmaWarmup::None)
            .init(&module);

        let snapshot_before = ema.model();

        let updated_module = WithRunning {
            bn: BatchNormConfig::new(2).init(&device),
            counter: RunningState::with_id(
                module.counter.value().let_id(&module.counter),
                Tensor::from_floats([11.0, 22.0], &device),
            ),
        };
        ema.update(&updated_module);

        // `snapshot_before` must remain [1.0, 2.0] while `ema.model()` reflects [6.0, 12.0].
        snapshot_before
            .counter
            .value()
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([1.0_f32, 2.0]), Tolerance::absolute(1e-5));
        ema.model()
            .counter
            .value()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([6.0_f32, 12.0]),
                Tolerance::absolute(1e-5),
            );
    }

    trait RunningStateIdExt {
        fn let_id(&self, state: &RunningState<Tensor<1>>) -> ParamId;
    }

    impl RunningStateIdExt for Tensor<1> {
        fn let_id(&self, state: &RunningState<Tensor<1>>) -> ParamId {
            struct IdVisitor(Option<ParamId>);
            impl ModuleVisitor for IdVisitor {
                fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                    self.0 = Some(param.id);
                }
            }
            let mut v = IdVisitor(None);
            state.visit(&mut v);
            v.0.unwrap()
        }
    }

    #[test]
    fn record_bytes_and_file_round_trip_preserve_weights_and_warmup_step() {
        let device = Device::default().autodiff();
        let mut model = make_linear([[1.0, 2.0], [3.0, 4.0]], [0.5, -0.5], &device);
        let mut sgd = SgdConfig::new().init();

        let config = ExponentialMovingAverageConfig::new().with_decay(0.95);
        let mut ema = config.init(&model);

        let x = Tensor::<2>::from_floats([[1.0, 0.5], [-0.5, 2.0]], &device);
        for _ in 0..3 {
            let loss = model.forward(x.clone()).mean();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = sgd.step(0.1, model, grads);
            ema.update(&model);
        }

        assert_eq!(ema.step(), 3);
        let bytes = ema.into_bytes().unwrap();

        // Restore into a freshly initialized EMA tracker and verify step + next update match.
        let fresh_model = LinearConfig::new(2, 2).init(&device);
        let mut restored = config.init(&fresh_model).from_bytes(bytes.clone()).unwrap();
        assert_eq!(restored.step(), 3);
        assert!((restored.effective_decay() - ema.effective_decay()).abs() < 1e-12);

        // Next optimizer + EMA update must produce identical shadow parameters.
        let loss = model.forward(x).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = sgd.step(0.1, model, grads);

        ema.update(&model);
        restored.update(&model);

        ema.model()
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &restored.model().weight.val().into_data(),
                Tolerance::absolute(1e-6),
            );

        // The serialized EMA burnpack bytes are also directly loadable as a ModuleRecord.
        let module_record = ModuleRecord::from_bytes(bytes).unwrap();
        let standalone = LinearConfig::new(2, 2)
            .init(&Device::default())
            .load_record(module_record);
        assert_eq!(standalone.weight.val().dims(), [2, 2]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn save_and_load_file_compatible_with_module_load_file() {
        let device = Device::default();
        let model = make_linear([[3.0, 6.0], [9.0, 12.0]], [1.0, 2.0], &device);
        let mut ema = ExponentialMovingAverageConfig::new()
            .with_decay(0.5)
            .with_warmup(EmaWarmup::None)
            .init(&model);
        let stepped = make_linear([[5.0, 10.0], [15.0, 20.0]], [3.0, 4.0], &device);
        ema.update(&stepped);

        let path = std::env::temp_dir().join(alloc::format!(
            "burn_optim_ema_{}_checkpoint.bpk",
            std::process::id()
        ));

        ema.save(&path).unwrap();

        let loaded_ema = ExponentialMovingAverageConfig::new()
            .with_decay(0.5)
            .with_warmup(EmaWarmup::None)
            .init(&LinearConfig::new(2, 2).init(&device))
            .load(&path)
            .unwrap();
        assert_eq!(loaded_ema.step(), 1);
        loaded_ema
            .model()
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &ema.model().weight.val().into_data(),
                Tolerance::absolute(1e-6),
            );

        // Directly load the same .bpk file into a Module via Module::load_file.
        let direct_module = LinearConfig::new(2, 2).init(&device).load_file(&path);
        direct_module
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &ema.model().weight.val().into_data(),
                Tolerance::absolute(1e-6),
            );

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[should_panic(expected = "EMA decay must be finite and within [0.0, 1.0]")]
    fn invalid_decay_panics() {
        let device = Device::default();
        let model = LinearConfig::new(2, 2).init(&device);
        let _ = ExponentialMovingAverageConfig::new()
            .with_decay(1.5)
            .init(&model);
    }

    #[test]
    #[should_panic(expected = "EmaWarmup::Linear warmup must be finite and >= 1.0")]
    fn invalid_linear_warmup_panics() {
        let device = Device::default();
        let model = LinearConfig::new(2, 2).init(&device);
        let _ = ExponentialMovingAverageConfig::new()
            .with_warmup(EmaWarmup::Linear { warmup: 0.5 })
            .init(&model);
    }
}

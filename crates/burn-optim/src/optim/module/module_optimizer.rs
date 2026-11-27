use burn_core as burn;
use burn_core::module::ParamGroup;

use super::global_norm::GlobalNorm;
use super::{ModuleLearningRate, Optimizer, ParameterContext};
use crate::{
    DynOptimizer, DynState, MultiGradientsParams, OptimizerRecord, RecordTensor, StateSink,
    StateSource, grad_clipping::GradientClipping, optim::GradientsParams, optim::state::join_path,
};

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::{vec, vec::Vec};
use burn::module::{Module, ModuleMapper, Param, ParamId};
use burn::store::RecordError;
use burn::tensor::{Bytes, Device, Tensor};
use hashbrown::HashMap;

/// Scalar key (per parameter) under which the parameter's state rank is persisted.
///
/// Reserved: a custom [`Optimizer::State`](crate::Optimizer::State) must not have a top-level
/// scalar field named `__rank`, as it would collide with this key in the record.
const RANK_KEY: &str = "__rank";

#[derive(Clone)]
struct OptimizerGroup {
    group: ParamGroup,
    optim: Arc<dyn DynOptimizer>,
    grad_clipping: Option<GradientClipping>,
}

fn matching_group(groups: &[OptimizerGroup], id: ParamId, path: Option<&str>) -> usize {
    groups
        .iter()
        .rposition(|g| g.group.matches(&id, path))
        .expect("Should match at least one parameter group.")
}

#[derive(Clone)]
struct OptimizationContext {
    // Groups are append-only. Cache the match while using its current clipping configuration.
    group_index: usize,
    path: Option<String>,
    state: DynState,
}

/// Optimizes a whole module by applying a per-parameter [`Optimizer`] to each of its parameters.
///
/// It is non-generic over the module and optimizer: any `O: Optimizer` is type-erased behind a
/// dynamic optimizer, and per-parameter states are kept as type-erased states keyed by
/// [`ParamId`](burn::module::ParamId). Build one with `optimizer.into()` or
/// `OptimizerConfig::init()`.
///
/// It is possible to use different optimizers for different parameters. To do so, use the
/// [ModuleOptimizer::with_group] function to add an optimizer for all parameters matching the
/// provided group.
#[derive(Clone)]
pub struct ModuleOptimizer {
    optimizers: Vec<OptimizerGroup>,
    param_context: HashMap<ParamId, OptimizationContext>,
}

impl<O> From<O> for ModuleOptimizer
where
    O: Optimizer,
{
    fn from(optim: O) -> Self {
        Self {
            param_context: HashMap::new(),
            optimizers: vec![OptimizerGroup {
                group: ParamGroup::all(),
                optim: Arc::new(optim),
                grad_clipping: None,
            }],
        }
    }
}

impl ModuleOptimizer {
    /// Check if the optimizer has gradient clipping.
    /// If there are multiple optimizers, checks if any group has gradient clipping.
    pub fn has_gradient_clipping(&self) -> bool {
        self.optimizers.iter().any(|g| g.grad_clipping.is_some())
    }

    /// Returns the clipping set by [`Self::with_grad_clipping`], including a module-wide norm limit.
    pub fn grad_clipping(&self) -> Option<&GradientClipping> {
        self.optimizers
            .first()
            .expect("Should have at least one optimizer")
            .grad_clipping
            .as_ref()
    }

    /// Sets clipping for optimizer updates. [`GradientClipping::GlobalNorm`] limits the joint L2
    /// norm across all parameter groups after gradient accumulation. Use `None` for clipping
    /// in every [`with_group`](Self::with_group) when this mode is selected.
    /// [`GradientClipping::Norm`] and [`GradientClipping::Value`] apply only to the fallback group.
    ///
    /// # Panics
    ///
    /// Panics if the global norm limit is negative or NaN, or if an added group has its own
    /// clipping configured when selecting [`GradientClipping::GlobalNorm`].
    pub fn with_grad_clipping(mut self, gradient_clipping: GradientClipping) -> Self {
        gradient_clipping.validate_global_limit();
        if matches!(gradient_clipping, GradientClipping::GlobalNorm(_)) {
            assert!(
                self.optimizers[1..]
                    .iter()
                    .all(|g| g.grad_clipping.is_none()),
                "GlobalNorm requires all parameter groups to use None for clipping"
            );
        }
        self.optimizers[0].grad_clipping = Some(gradient_clipping);
        self
    }

    fn step_common<M: Module>(
        &mut self,
        lr_policy: ModuleLearningRate,
        module: M,
        grads: GradAdaptor,
    ) -> M {
        let (mut grads, global_norm) = match self.grad_clipping() {
            Some(GradientClipping::GlobalNorm(max_norm)) => {
                let (grads, norm) = GlobalNorm::prepare(&module, grads, *max_norm);
                (grads.into(), norm)
            }
            _ => (grads, None),
        };
        module.map(&mut ModuleOptimizerMapper {
            path: Vec::new(),
            optimizer_groups: &self.optimizers,
            states: &mut self.param_context,
            grads: &mut grads,
            global_norm,
            lr_module: lr_policy,
        })
    }

    /// Adds an optimizer specific to a parameter group.
    ///
    /// Parameters matching this group will be optimized using the provided optimizer
    /// and gradient clipping configuration.
    ///
    /// ### Matching Rules
    /// * **Precedence:** If a parameter matches multiple groups, the *last* group added takes precedence.
    /// * **Fallback:** The first optimizer added must match all parameters to act as a global fallback.
    ///
    /// ### Side Effects
    /// * **State Reset:** Adding a new group will reset any existing optimizer states for parameters
    ///   that match the new group.
    ///
    /// # Panics
    ///
    /// Panics for [`GradientClipping::GlobalNorm`]. Set it with [`Self::with_grad_clipping`]
    /// to include gradients from every group. Panics if group clipping is supplied while
    /// the module optimizer uses `GlobalNorm`.
    pub fn with_group<O>(
        mut self,
        group: ParamGroup,
        optim: O,
        grad_clipping: Option<GradientClipping>,
    ) -> Self
    where
        O: DynOptimizer + 'static,
    {
        assert!(
            !matches!(grad_clipping, Some(GradientClipping::GlobalNorm(_))),
            "Set GlobalNorm on the module optimizer with with_grad_clipping"
        );
        assert!(
            grad_clipping.is_none()
                || !matches!(self.grad_clipping(), Some(GradientClipping::GlobalNorm(_))),
            "GlobalNorm requires all parameter groups to use None for clipping"
        );
        self.optimizers.push(OptimizerGroup {
            group: group.clone(),
            optim: Arc::new(optim),
            grad_clipping,
        });
        self.param_context
            .retain(|id, param_state| !group.matches(id, param_state.path.as_deref()));
        self
    }
}

impl ModuleOptimizer {
    /// Update the `module` parameters with the given `gradients`, advancing the optimizer state.
    ///
    /// # Panics
    ///
    /// Panics if `GlobalNorm` receives a quantized parameter gradient.
    pub fn step<M: Module>(
        &mut self,
        lr_module: impl Into<ModuleLearningRate>,
        module: M,
        grads: GradientsParams,
    ) -> M {
        self.step_common(lr_module.into(), module, grads.into())
    }

    /// Like [`step`](Self::step), but accumulating gradients sourced from multiple devices.
    ///
    /// With `GlobalNorm`, each parameter's running sum must fit the accumulation dtype (F64 for
    /// F64 gradients, F32 otherwise). Clipping takes place after accumulation.
    ///
    /// # Panics
    ///
    /// Panics if `GlobalNorm` receives a quantized gradient contribution.
    pub fn step_multi<M: Module>(
        &mut self,
        lr_module: impl Into<ModuleLearningRate>,
        module: M,
        grads: MultiGradientsParams,
    ) -> M {
        self.step_common(lr_module.into(), module, grads.into())
    }

    /// Decompose the optimizer state into a serializable [`OptimizerRecord`].
    pub fn to_record(&self) -> OptimizerRecord {
        let mut tensors = Vec::new();
        let mut scalars = BTreeMap::new();
        let mut paths = BTreeMap::new();

        for (id, param_state) in self.param_context.iter() {
            let prefix = id.val().to_string();
            let mut sink = StateSink::default();
            self.optimizers[param_state.group_index]
                .optim
                .state_flatten(&prefix, &param_state.state, &mut sink);

            // Persist the parameter rank explicitly so the state can be reconstructed even when it
            // carries no tensors, and without inferring the rank from tensor shapes.
            scalars.insert(
                join_path(&prefix, RANK_KEY),
                burn_pack::Scalar::from(param_state.state.rank()),
            );
            // Save parameter path to be able to match to the right group when loading.
            if let Some(path) = &param_state.path {
                paths.insert(prefix, path.clone());
            }

            for (name, data) in sink.tensors {
                tensors.push(RecordTensor {
                    name,
                    param_id: Some(id.val()),
                    data,
                });
            }
            for (name, value) in sink.scalars {
                scalars.insert(name, value);
            }
        }

        OptimizerRecord {
            tensors,
            scalars,
            paths,
        }
    }

    /// Load the optimizer state from an [`OptimizerRecord`].
    ///
    /// State tensors are materialized on the default device, which must support their recorded
    /// numeric dtypes. The next [`step`](ModuleOptimizer::step) moves each parameter's state to
    /// its gradient's device, which must also support the saved dtypes.
    pub fn load_record(mut self, record: OptimizerRecord) -> Self {
        let device = Device::default();
        let mut ranks: BTreeMap<u64, usize> = BTreeMap::new();
        let mut paths: BTreeMap<u64, String> = BTreeMap::new();

        // Recover each parameter's rank from its persisted `__rank` scalar (authoritative). Keys
        // are `"{param_id}.__rank"`, so strip the dotted suffix to recover the id.
        let suffix = alloc::format!(".{RANK_KEY}");
        for (name, value) in record.scalars.iter() {
            if let Some(id_str) = name.strip_suffix(&suffix)
                && let (Ok(id), Ok(rank)) = (id_str.parse::<u64>(), usize::try_from(*value))
            {
                ranks.insert(id, rank);
            }
        }

        for (name, path) in record.paths.iter() {
            if let Ok(id) = name.parse::<u64>() {
                paths.insert(id, path.to_string());
            }
        }

        let mut source = StateSource::new(record.scalars);

        for tensor in record.tensors {
            let RecordTensor {
                name,
                param_id,
                data,
            } = tensor;
            let id = param_id.expect("Optimizer record tensors should carry a parameter id.");
            // Fall back to inferring rank from a tensor shape if no `__rank` scalar was present.
            ranks.entry(id).or_insert(data.rank());
            source.insert_tensor(name, data);
        }

        let mut states = HashMap::new();
        for (id, rank) in ranks {
            let prefix = id.to_string();
            let path = paths.get(&id);
            let group_index = matching_group(&self.optimizers, id.into(), path.map(String::as_str));
            let optim = &self.optimizers[group_index].optim;
            // Skip parameters whose state can't be reconstructed (truncated/foreign record); they
            // are re-initialized lazily on the next step rather than aborting the load.
            if let Some(state) = optim.state_unflatten(rank, &prefix, &mut source, &device) {
                states.insert(
                    ParamId::from(id),
                    OptimizationContext {
                        group_index,
                        path: path.cloned(),
                        state,
                    },
                );
            }
        }

        self.param_context = states;
        self
    }

    /// Serialize the optimizer state to an in-memory burnpack byte buffer.
    pub fn into_bytes(&self) -> Result<Bytes, RecordError> {
        self.to_record().into_bytes()
    }

    /// Load the optimizer state from an in-memory burnpack byte buffer.
    pub fn from_bytes(self, bytes: Bytes) -> Result<Self, RecordError> {
        Ok(self.load_record(OptimizerRecord::from_bytes(bytes)?))
    }

    /// Save the optimizer state to a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), RecordError> {
        self.to_record().save(path)
    }

    /// Load the optimizer state from a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn load<P: AsRef<std::path::Path>>(self, path: P) -> Result<Self, RecordError> {
        Ok(self.load_record(OptimizerRecord::load(path)?))
    }
}

/// Wrapper to unify the `remove` method for [GradientsParams] and [MultiGradientsParams].
pub enum GradAdaptor {
    /// Wrapper for [`GradientsParams`].
    Single(GradientsParams),

    /// Wrapper for [`MultiGradientsParams`].
    Multi(MultiGradientsParams),
}

impl From<GradientsParams> for GradAdaptor {
    fn from(grads: GradientsParams) -> Self {
        Self::Single(grads)
    }
}

impl From<MultiGradientsParams> for GradAdaptor {
    fn from(grads: MultiGradientsParams) -> Self {
        Self::Multi(grads)
    }
}

impl GradAdaptor {
    /// Remove a gradient parameter by ID.
    ///
    /// # Returns
    /// Maybe the (tensor, device) pair.
    pub fn remove<const D: usize>(&mut self, id: ParamId) -> Option<(Tensor<D>, Device)> {
        match self {
            GradAdaptor::Single(grads) => grads.remove(id).map(|t| {
                let device = t.device();
                (t, device)
            }),
            GradAdaptor::Multi(grads) => grads.remove(id),
        }
    }
}

struct ModuleOptimizerMapper<'a> {
    path: Vec<String>,
    optimizer_groups: &'a [OptimizerGroup],
    global_norm: Option<GlobalNorm>,
    states: &'a mut HashMap<ParamId, OptimizationContext>,
    grads: &'a mut GradAdaptor,
    lr_module: ModuleLearningRate,
}

impl ModuleMapper for ModuleOptimizerMapper<'_> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let grad = self.grads.remove::<D>(id);

        let tensor = if let Some((grad, device)) = grad {
            let context = ParameterContext::capture(&tensor);

            let tensor = if tensor.device() != device {
                tensor.to_device(&device)
            } else {
                tensor
            };

            let path = self.path.join(".");
            let (group_index, existing_dyn_state) = match self.states.remove(&id) {
                Some(context) => (context.group_index, Some(context.state)),
                None => (matching_group(self.optimizer_groups, id, Some(&path)), None),
            };
            let group = &self.optimizer_groups[group_index];
            let optim = &group.optim;

            debug_assert_eq!(
                grad.device(),
                device,
                "The gradient is on the provided device"
            );
            let clipped_grad = match (&self.global_norm, &group.grad_clipping) {
                (Some(global_norm), _) => global_norm.clip(id, grad, &device),
                (None, Some(clipping)) => clipping.clip_gradient(grad),
                (None, None) => grad,
            };

            debug_assert_eq!(
                tensor.device(),
                device,
                "Tensor and gradients are on the same device."
            );

            let lr = self.lr_module.lr_from_param(id, Some(path.as_str()));
            let (tensor, state) = optim.step_dyn(
                D,
                lr,
                tensor.inner().into_bridge(),
                clipped_grad.into_bridge(),
                existing_dyn_state.map(|s| optim.to_device_dyn(s, &device)),
            );

            if let Some(state) = state {
                self.states.insert(
                    id,
                    OptimizationContext {
                        group_index,
                        path: Some(path),
                        state,
                    },
                );
            }

            context.restore(Tensor::from_bridge(tensor), id)
        } else {
            tensor
        };

        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdamConfig, GradientsParams, SgdConfig,
        lr_scheduler::module_lr_scheduler::ModuleLearningRate,
    };
    use burn::module::ParamGroup;
    use burn::tensor::{DType, Distribution, Tensor, Tolerance};
    use burn_derive::Module;
    use burn_nn::{Linear, LinearConfig};

    #[derive(Module, Debug)]
    struct TwoLayerModel {
        layer_a: Linear,
        layer_b: Linear,
    }

    fn make_model(device: &Device) -> TwoLayerModel {
        TwoLayerModel {
            layer_a: LinearConfig::new(4, 4).init(device),
            layer_b: LinearConfig::new(4, 4).init(device),
        }
    }

    fn make_grads(model: &TwoLayerModel, x: Tensor<2>) -> GradientsParams {
        let out = model.layer_a.forward(x.clone()) + model.layer_b.forward(x);
        GradientsParams::from_grads(out.mean().backward(), model)
    }

    fn lr() -> ModuleLearningRate {
        ModuleLearningRate::from(0.01_f64)
    }

    /// A parameter trains under the checkpointing strategy of the device it
    /// was created on. The update leaves the tape and comes back, and it must
    /// come back with that strategy, or the next step's operations merge
    /// parameters of two strategies and refuse.
    #[test]
    fn step_keeps_the_gradient_checkpointing_strategy() {
        let device = Device::default().autodiff().gradient_checkpointing();
        let model = make_model(&device);
        let mut optim = sgd();

        let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);
        let model = optim.step(lr(), model.clone(), make_grads(&model, x.clone()));

        assert_eq!(
            model.layer_a.weight.val().gradient_checkpointing_strategy(),
            device.gradient_checkpointing_strategy()
        );
        // Panics without the fix: the updated parameter and the input meet
        // under two strategies.
        let _ = optim.step(lr(), model.clone(), make_grads(&model, x));
    }

    fn sgd() -> ModuleOptimizer {
        SgdConfig::new().init()
    }

    /// to_record / load_record must fully preserve a stateful optimizer's internal state so that
    /// a step on the restored optimizer is numerically identical to one taken on the original.
    #[test]
    fn default_optimizer_state_survives_round_trip() {
        let device = Device::default().autodiff();
        let mut model = make_model(&device);
        let mut optim: ModuleOptimizer = AdamConfig::new().init();

        for _ in 0..3 {
            let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);
            model = optim.step(lr(), model.clone(), make_grads(&model, x));
        }

        let record = optim.to_record();
        let mut reloaded: ModuleOptimizer = AdamConfig::new().init().load_record(record);

        let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);
        let grads_a = make_grads(&model, x.clone());
        let grads_b = make_grads(&model, x);
        let from_orig = optim.step(lr(), model.clone(), grads_a);
        let from_reload = reloaded.step(lr(), model, grads_b);

        from_orig
            .layer_a
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &from_reload.layer_a.weight.val().into_data(),
                Tolerance::absolute(1e-6),
            );
    }

    /// The paths saved in OptimizerRecord enable load_record to route each parameter to the
    /// correct group optimizer. A step on the restored optimizer must be numerically identical
    /// to one on the original — for both the group (Adam) and the default (SGD) optimizer.
    #[test]
    fn group_optimizer_routes_correctly_after_record_round_trip() {
        let device = Device::default().autodiff();
        let mut model = make_model(&device);

        let make_optim = || {
            sgd().with_group(
                ParamGroup::from_predicate("layer_a"),
                AdamConfig::new().build(),
                None,
            )
        };
        let mut optim = make_optim();

        for _ in 0..3 {
            let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);
            model = optim.step(lr(), model.clone(), make_grads(&model, x));
        }

        let record = optim.to_record();
        let mut reloaded = make_optim().load_record(record);

        let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);
        let grads_a = make_grads(&model, x.clone());
        let grads_b = make_grads(&model, x);
        let from_orig = optim.step(lr(), model.clone(), grads_a);
        let from_reload = reloaded.step(lr(), model, grads_b);

        from_orig
            .layer_a
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &from_reload.layer_a.weight.val().into_data(),
                Tolerance::absolute(1e-6),
            );
        from_orig
            .layer_b
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &from_reload.layer_b.weight.val().into_data(),
                Tolerance::absolute(1e-6),
            );
    }

    /// Adding a group after training must clear accumulated state for matching params
    /// while leaving state for non-matching params untouched.
    #[test]
    fn with_group_clears_state_for_matching_params() {
        let device = Device::default().autodiff();
        let mut model = make_model(&device);
        let mut optim: ModuleOptimizer = AdamConfig::new().init();

        for _ in 0..2 {
            let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);
            model = optim.step(lr(), model.clone(), make_grads(&model, x));
        }

        // Switch layer_a to SGD. Its Adam state must be cleared.
        optim = optim.with_group(
            ParamGroup::from_predicate("layer_a"),
            SgdConfig::new().build(),
            None,
        );

        let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);
        _ = optim.step(lr(), model.clone(), make_grads(&model, x));

        let record = optim.to_record();

        let time_key_count = record.scalars.keys().filter(|k| k.contains("time")).count();
        assert_eq!(
            time_key_count, 2,
            "only layer_b params should carry Adam's time scalar after group switch"
        );
    }

    #[test]
    fn group_grad_clipping_applies_only_to_matching_params() {
        let device = Device::default().autodiff();
        let mut model = make_model(&device);

        let lr_value = 0.01;
        let threshold = 0.01_f32;
        let bound = (lr_value * threshold as f64) as f32;

        let mut optim = sgd().with_group(
            ParamGroup::from_predicate("layer_a"),
            SgdConfig::new().build(),
            Some(GradientClipping::Value(threshold)),
        );

        // Large inputs produce gradients that clearly exceed the clipping threshold.
        let x = Tensor::<2>::random([2, 4], Distribution::Uniform(100.0, 200.0), &device);
        let weight_a_before = model.layer_a.weight.val();
        let weight_b_before = model.layer_b.weight.val();

        model = optim.step(
            ModuleLearningRate::from(lr_value),
            model.clone(),
            make_grads(&model, x),
        );

        let diff_a = weight_a_before - model.layer_a.weight.val();
        let diff_b = weight_b_before - model.layer_b.weight.val();

        let eps = 1e-6_f32;
        for value in diff_a.into_data().iter::<f32>() {
            assert!(
                value.abs() <= bound + eps,
                "layer_a's group grad_clipping should keep every update within lr * threshold"
            );
        }
        assert!(
            diff_b
                .into_data()
                .iter::<f32>()
                .any(|value| value.abs() > bound + eps),
            "layer_b uses the default optimizer and should not be clipped"
        );
    }

    fn scalar_params(dtype: DType) -> [Param<Tensor<1>>; 2] {
        let device = Device::default().autodiff();
        core::array::from_fn(|_| Param::from_tensor(Tensor::zeros([1], (&device, dtype))))
    }

    fn scalar_grads(params: &[Param<Tensor<1>>; 2], values: [f32; 2]) -> GradientsParams {
        let mut grads = GradientsParams::new();
        for (param, value) in params.iter().zip(values) {
            grads.register(
                param.id,
                Tensor::<1>::from_floats([value], &Device::default()).cast(param.val().dtype()),
            );
        }
        grads
    }

    fn assert_params(params: &[Param<Tensor<1>>; 2], expected: [f32; 2]) {
        for (param, value) in params.iter().zip(expected) {
            let actual = param.val().into_scalar::<f32>();
            assert!(
                (actual - value).abs() < 1e-5,
                "expected {value}, got {actual}"
            );
        }
    }

    #[test]
    fn global_norm_spans_parameter_groups() {
        let model = scalar_params(DType::F32);
        let mut optim = sgd()
            .with_grad_clipping(GradientClipping::GlobalNorm(4.5))
            .with_group(
                ParamGroup::from_ids(vec![model[1].id]),
                SgdConfig::new().build(),
                None,
            );
        let grads = scalar_grads(&model, [3.0, 4.0]);
        assert_params(&optim.step(1.0, model, grads), [-2.7, -3.6]);
    }

    #[test]
    #[should_panic(expected = "GlobalNorm requires all parameter groups to use None for clipping")]
    fn global_norm_rejects_adding_a_group_with_clipping() {
        sgd()
            .with_grad_clipping(GradientClipping::GlobalNorm(1.0))
            .with_group(
                ParamGroup::all(),
                SgdConfig::new().build(),
                Some(GradientClipping::Value(0.5)),
            );
    }

    #[test]
    #[should_panic(expected = "GlobalNorm requires all parameter groups to use None for clipping")]
    fn global_norm_rejects_existing_group_clipping() {
        sgd()
            .with_group(
                ParamGroup::all(),
                SgdConfig::new().build(),
                Some(GradientClipping::Norm(0.5)),
            )
            .with_grad_clipping(GradientClipping::GlobalNorm(1.0));
    }

    #[test]
    fn global_norm_scales_gradients_before_momentum() {
        let model = scalar_params(DType::F32);
        let momentum = crate::momentum::MomentumConfig::new().with_momentum(0.5);
        let mut optim = SgdConfig::new()
            .with_momentum(Some(momentum))
            .init()
            .with_grad_clipping(GradientClipping::GlobalNorm(4.5));
        let grads = scalar_grads(&model, [3.0, 4.0]);
        let model = optim.step(0.0, model, grads);

        // Suppress the new gradients so this update exposes only the previous velocity.
        let grads = scalar_grads(&model, [10.0, 10.0]);
        let model = optim
            .with_grad_clipping(GradientClipping::Value(0.0))
            .step(1.0, model, grads);
        assert_params(&model, [-1.35, -1.8]);
    }

    #[test]
    fn global_norm_uses_accumulated_gradients_in_step_multi() {
        let model = scalar_params(DType::F32);
        let mut optim = SgdConfig::new()
            .with_gradient_clipping(Some(
                crate::grad_clipping::GradientClippingConfig::GlobalNorm(4.5),
            ))
            .init();
        let contributions = MultiGradientsParams {
            grads: [[8.0, 1.0], [-5.0, 3.0]]
                .map(|values| (scalar_grads(&model, values), Device::default()))
                .into(),
        };
        let single = optim
            .clone()
            .step(1.0, model.clone(), scalar_grads(&model, [3.0, 4.0]));
        let multi = optim.step_multi(1.0, model, contributions);
        assert_params(&single, [-2.7, -3.6]);
        assert_params(&multi, single.map(|p| p.val().into_scalar::<f32>()));
    }

    fn quantized_grad() -> Tensor<1> {
        let device = Device::default();
        let scheme = device.settings().quantization.scheme;
        Tensor::from_floats([1., 2., 3., 4.], &device).quantize_dynamic(&scheme)
    }

    #[test]
    #[should_panic(expected = "GlobalNorm requires floating-point gradients")]
    fn global_norm_rejects_quantized_gradient_in_step() {
        let device = Device::default().autodiff();
        let param = Param::from_tensor(Tensor::<1>::zeros([4], &device));
        let mut grads = GradientsParams::new();
        grads.register(param.id, quantized_grad());
        sgd()
            .with_grad_clipping(GradientClipping::GlobalNorm(1.))
            .step(1., param, grads);
    }

    #[test]
    #[should_panic(expected = "GlobalNorm requires floating-point gradients")]
    fn global_norm_rejects_quantized_contribution_before_accumulation() {
        let device = Device::default();
        let param = Param::from_tensor(Tensor::<1>::zeros([4], &device.clone().autodiff()));
        let contributions = [Tensor::ones([4], &device), quantized_grad()].map(|grad| {
            let mut grads = GradientsParams::new();
            grads.register(param.id, grad);
            (grads, device.clone())
        });
        sgd()
            .with_grad_clipping(GradientClipping::GlobalNorm(1.))
            .step_multi(
                1.,
                param,
                MultiGradientsParams {
                    grads: contributions.into(),
                },
            );
    }

    #[test]
    fn global_norm_handles_zero_and_missing_gradients() {
        let model = scalar_params(DType::F32);
        let mut optim = sgd().with_grad_clipping(GradientClipping::GlobalNorm(0.0));
        let mut grads = GradientsParams::new();
        grads.register(model[0].id, Tensor::<1>::zeros([1], &Device::default()));
        let updated = optim.step(1.0, model, grads);
        assert_params(&updated, [0.0, 0.0]);
        let paths = optim.to_record().paths;
        assert_eq!(paths.len(), 1);
        assert!(paths.contains_key(&updated[0].id.val().to_string()));
        let updated = optim.step(1.0, updated, GradientsParams::new());
        assert_params(&updated, [0.0, 0.0]);
    }

    #[test]
    fn global_norm_counts_only_distinct_module_gradients() {
        let params = scalar_params(DType::F32);
        let mut grads = scalar_grads(&params, [3.0, 4.0]);
        grads.register(
            ParamId::new(),
            Tensor::<1>::from_floats([100.0], &Device::default()),
        );
        let model = [params[0].clone(), params[0].clone(), params[1].clone()];
        let updated = sgd()
            .with_grad_clipping(GradientClipping::GlobalNorm(5.5))
            .step(1.0, model, grads);
        let [first, _, last] = updated;
        assert_params(&[first, last], [-3.0, -4.0]);
    }

    #[test]
    fn global_norm_squares_fp16_gradients_in_fp32() {
        let model = scalar_params(DType::F16);
        let grads = scalar_grads(&model, [300.0, 400.0]);
        let updated = sgd()
            .with_grad_clipping(GradientClipping::GlobalNorm(50.0))
            .step(1.0, model, grads);
        assert_params(&updated, [-30.0, -40.0]);
        assert_eq!(updated[0].val().dtype(), DType::F16);
    }

    #[test]
    fn global_norm_widens_contributions_before_accumulation() {
        let model = scalar_params(DType::F16);
        let grads = MultiGradientsParams {
            grads: [[40_000., 20_000.]; 2]
                .map(|values| (scalar_grads(&model, values), Device::default()))
                .into(),
        };
        let mut optim = sgd().with_grad_clipping(GradientClipping::GlobalNorm(5f32.sqrt()));
        let updated = optim.step_multi(1., model, grads);
        assert_params(&updated, [-2., -1.]);
        assert_eq!(updated[0].val().dtype(), DType::F16);
    }

    #[test]
    fn global_norm_combines_large_finite_gradients() {
        let model = scalar_params(DType::F32);
        let grads = scalar_grads(&model, [3e20, 4e20]);
        let mut optim = sgd().with_grad_clipping(GradientClipping::GlobalNorm(1.));
        assert_params(&optim.step(1., model, grads), [-0.6, -0.8]);
    }

    #[test]
    #[should_panic(expected = "GlobalNorm requires a nonnegative limit")]
    fn global_norm_rejects_nan_limit() {
        sgd().with_grad_clipping(GradientClipping::GlobalNorm(f32::NAN));
    }

    #[test]
    #[should_panic(expected = "Set GlobalNorm on the module optimizer")]
    fn global_norm_rejects_group_level_clipping() {
        sgd().with_group(
            ParamGroup::all(),
            SgdConfig::new().build(),
            Some(GradientClipping::GlobalNorm(1.)),
        );
    }

    /// An OptimizerRecord with empty paths map must load cleanly.
    /// Parameters default to the default optimizer.
    #[test]
    fn record_without_paths_loads_without_panic() {
        let device = Device::default().autodiff();
        let mut model = make_model(&device);
        let mut optim: ModuleOptimizer = AdamConfig::new().init();

        let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);
        model = optim.step(lr(), model.clone(), make_grads(&model, x));

        // Simulate a record with empty paths.
        let mut record = optim.to_record();
        record.paths.clear();

        let mut optim_loaded: ModuleOptimizer = AdamConfig::new().init().load_record(record);

        let x = Tensor::<2>::random([2, 4], Distribution::Default, &device);
        let grads_a = make_grads(&model, x.clone());
        let grads_b = make_grads(&model, x);
        let from_orig = optim.step(lr(), model.clone(), grads_a);
        let from_reload = optim_loaded.step(lr(), model, grads_b);

        from_orig
            .layer_a
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &from_reload.layer_a.weight.val().into_data(),
                Tolerance::absolute(1e-6),
            );
        from_orig
            .layer_b
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &from_reload.layer_b.weight.val().into_data(),
                Tolerance::absolute(1e-6),
            );
    }
}

use burn_core as burn;
use burn_core::module::ParamGroup;

use super::Optimizer;
use crate::lr_scheduler::policy::LrPolicy;
use crate::{
    DynOptimizer, DynState, MultiGradientsParams, OptimizerRecord, StateSink, StateSource,
    grad_clipping::GradientClipping, optim::GradientsParams, optim::state::join_path,
};

use alloc::collections::BTreeMap;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use burn::module::{AutodiffModule, ModuleMapper, Param, ParamId};
use burn::store::RecordError;
use burn::tensor::{Bytes, Device, Tensor, TensorData};
use hashbrown::HashMap;

/// Scalar key (per parameter) under which the parameter's state rank is persisted.
///
/// Reserved: a custom [`Optimizer::State`](crate::Optimizer::State) must not have a top-level
/// scalar field named `__rank`, as it would collide with this key in the record.
const RANK_KEY: &str = "__rank";

#[derive(Clone)]
struct OptimizerGroup {
    group: ParamGroup,
    grad_clipping: Option<GradientClipping>,
    optim: Arc<dyn DynOptimizer>,
}

/// Keep a reference to the optimizer to avoid matching every step.
#[derive(Clone)]
struct ParamOptimizationState {
    optim: Arc<dyn DynOptimizer>,
    grad_clipping: Option<GradientClipping>,
    path: Option<String>,
    state: DynState,
}

/// Optimizes a whole module by applying a per-parameter [`Optimizer`] to each of its parameters.
///
/// It is non-generic over the module and optimizer: any `O: Optimizer` is type-erased behind a
/// dynamic optimizer, and per-parameter states are kept as type-erased states keyed by
/// [`ParamId`](burn::module::ParamId). Build one with `optimizer.into()` or
/// `OptimizerConfig::init()`.
#[derive(Clone)]
pub struct ModuleOptimizer {
    default_optim: Arc<dyn DynOptimizer>,
    optim_groups: Vec<OptimizerGroup>,
    param_state_map: HashMap<ParamId, ParamOptimizationState>,
    default_grad_clipping: Option<GradientClipping>,
}

impl<O> From<O> for ModuleOptimizer
where
    O: Optimizer,
{
    fn from(optim: O) -> Self {
        Self {
            default_optim: Arc::new(optim),
            param_state_map: HashMap::new(),
            default_grad_clipping: None,
            optim_groups: vec![],
        }
    }
}

impl ModuleOptimizer {
    /// Check if the optimizer has gradient clipping.
    pub fn has_gradient_clipping(&self) -> bool {
        self.default_grad_clipping.is_some()
    }

    /// Access the gradient clipping.
    pub fn grad_clipping(&self) -> Option<&GradientClipping> {
        self.default_grad_clipping.as_ref()
    }

    /// Sets the gradient clipping.
    ///
    /// # Arguments
    ///
    /// * `gradient_clipping` - The gradient clipping.
    ///
    /// # Returns
    ///
    /// The optimizer.
    pub fn with_grad_clipping(mut self, gradient_clipping: GradientClipping) -> Self {
        self.default_grad_clipping = Some(gradient_clipping);
        self
    }

    fn step_common<M: AutodiffModule>(
        &mut self,
        lr_policy: LrPolicy,
        module: M,
        mut grads: GradAdaptor,
    ) -> M {
        module.map(&mut ModuleOptimizerMapper::new(
            vec![],
            &self.default_optim,
            self.optim_groups.iter().map(|g| g).collect(),
            &mut self.param_state_map,
            &mut grads,
            lr_policy,
            self.default_grad_clipping.as_ref(),
        ))
    }

    /// Add a parameter group-specific optimizer. Parameters matching this group will be optimized
    /// using the provided optimizer. Existing states for parameters matching this group will be reset.
    pub fn with_group(
        mut self,
        group: ParamGroup,
        optim: impl Into<ModuleOptimizer>,
        grad_clipping: Option<GradientClipping>,
    ) -> Self {
        self.optim_groups.push(OptimizerGroup {
            group: group.clone(),
            optim: optim.into().default_optim,
            grad_clipping,
        });
        self.param_state_map.retain(|id, param_state| {
            !group.matches(id, param_state.path.as_ref().map(|p| p.as_str()))
        });
        self
    }
}

impl ModuleOptimizer {
    /// Update the `module` parameters with the given `gradients`, advancing the optimizer state.
    pub fn step<M: AutodiffModule>(
        &mut self,
        lr_policy: LrPolicy,
        module: M,
        grads: GradientsParams,
    ) -> M {
        self.step_common(lr_policy, module, grads.into())
    }

    /// Like [`step`](Self::step), but accumulating gradients sourced from multiple devices.
    pub fn step_multi<M: AutodiffModule>(
        &mut self,
        lr_policy: LrPolicy,
        module: M,
        grads: MultiGradientsParams,
    ) -> M {
        self.step_common(lr_policy, module, grads.into())
    }

    fn optim_from_param(
        &self,
        id: ParamId,
        path: Option<&str>,
    ) -> (&'_ Arc<dyn DynOptimizer>, Option<GradientClipping>) {
        self.optim_groups
            .iter()
            .filter_map(|val| {
                val.group
                    .matches(&id, path)
                    .then_some((&val.optim, val.grad_clipping.clone()))
            })
            .last()
            .unwrap_or((&self.default_optim, self.default_grad_clipping.clone()))
    }

    /// Decompose the optimizer state into a serializable [`OptimizerRecord`].
    pub fn to_record(&self) -> OptimizerRecord {
        let mut tensors = Vec::new();
        let mut scalars = BTreeMap::new();
        let mut paths = BTreeMap::new();

        for (id, param_state) in self.param_state_map.iter() {
            let prefix = id.val().to_string();
            let mut sink = StateSink::default();
            param_state
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
                tensors.push(burn_pack::Tensor::new(
                    name,
                    data.dtype,
                    data.shape,
                    Some(id.val()),
                    data.bytes,
                ));
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
    /// State tensors are materialized on the default device; no device argument is needed because
    /// each parameter's state is migrated to that parameter's (gradient's) device on the next
    /// [`step`](ModuleOptimizer::step) — see the `to_device` call in the step path. The load device
    /// is therefore irrelevant to correctness.
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
            let id = tensor
                .param_id
                .expect("Optimizer record tensors should carry a parameter id.");
            let name = tensor.name;
            let data = TensorData::from_bytes(tensor.bytes, tensor.shape, tensor.dtype);
            // Fall back to inferring rank from a tensor shape if no `__rank` scalar was present.
            ranks.entry(id).or_insert(data.shape.len());
            source.insert_tensor(name, data);
        }

        let mut states = HashMap::new();
        for (id, rank) in ranks {
            let prefix = id.to_string();
            let path = paths.get(&id);
            let (optim, grad_clipping) =
                self.optim_from_param(id.into(), path.map(|path| path.as_str()));
            // Skip parameters whose state can't be reconstructed (truncated/foreign record); they
            // are re-initialized lazily on the next step rather than aborting the load.
            if let Some(state) = optim.state_unflatten(rank, &prefix, &mut source, &device) {
                states.insert(
                    ParamId::from(id),
                    ParamOptimizationState {
                        optim: optim.clone(),
                        path: path.map(|p| p.clone()),
                        state,
                        grad_clipping,
                    },
                );
            }
        }

        self.param_state_map = states;
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

#[derive(new)]
struct ModuleOptimizerMapper<'a> {
    path: Vec<String>,
    optimizer: &'a Arc<dyn DynOptimizer>,
    optimizer_groups: Vec<&'a OptimizerGroup>,
    states: &'a mut HashMap<ParamId, ParamOptimizationState>,
    grads: &'a mut GradAdaptor,
    lr_policy: LrPolicy,
    grad_clipping: Option<&'a GradientClipping>,
}

impl ModuleOptimizerMapper<'_> {
    fn optimizer_from_param(
        &self,
        id: ParamId,
        path: Option<&str>,
    ) -> (Arc<dyn DynOptimizer>, Option<GradientClipping>) {
        self.optimizer_groups
            .iter()
            .filter_map(|val| {
                val.group
                    .matches(&id, path)
                    .then_some((val.optim.clone(), val.grad_clipping.clone()))
            })
            .last()
            .unwrap_or((self.optimizer.clone(), self.grad_clipping.cloned()))
    }
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
        let grad = self.grads.remove(id);

        let tensor = if let Some((grad, device)) = grad {
            let is_require_grad = tensor.is_require_grad();
            #[cfg(feature = "std")]
            let is_distributed = tensor.is_distributed();

            let entry = self.states.remove_entry(&id);
            let key = entry.as_ref().map(|(k, _)| *k);
            let tensor = if tensor.device() != device {
                tensor.to_device(&device)
            } else {
                tensor
            };

            let path = self.path.join(".");
            let (optim, grad_clipping, existing_dyn_state) = match entry.map(|(_, s)| s) {
                Some(ParamOptimizationState {
                    optim,
                    grad_clipping,
                    state,
                    ..
                }) => (optim, grad_clipping, Some(state)),
                None => {
                    let (optim, grad_clipping) =
                        self.optimizer_from_param(id, Some(path.as_str())).clone();
                    (optim, grad_clipping, None)
                }
            };

            debug_assert_eq!(
                grad.device(),
                device,
                "The gradient is on the provided device"
            );
            let clipped_grad: Tensor<D> = if let Some(g_clipping) = grad_clipping.as_ref() {
                g_clipping.clip_gradient(grad)
            } else {
                grad
            };

            debug_assert_eq!(
                tensor.device(),
                device,
                "Tensor and gradients are on the same device."
            );

            let lr = self.lr_policy.lr_from_param(id, Some(path.as_str()));
            let (tensor, state) = optim.step_dyn(
                D,
                lr,
                tensor.inner().into_bridge(),
                clipped_grad.into_bridge(),
                existing_dyn_state.map(|s| optim.to_device_dyn(s, &device)),
            );

            if let Some(state) = state {
                self.states.insert(
                    key.unwrap_or(id),
                    ParamOptimizationState {
                        optim,
                        path: Some(path),
                        state,
                        grad_clipping,
                    },
                );
            }

            let mut tensor = Tensor::from_inner(Tensor::from_bridge(tensor));

            if is_require_grad {
                tensor = tensor.require_grad();
            }
            #[cfg(feature = "std")]
            if is_distributed {
                tensor = tensor.set_distributed(id)
            }

            tensor
        } else {
            tensor
        };

        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdamConfig, GradientsParams, SgdConfig, lr_scheduler::policy::LrPolicy};
    use burn::module::ParamGroup;
    use burn::tensor::{Distribution, Tensor, Tolerance};
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

    fn lr() -> LrPolicy {
        LrPolicy::from(0.01_f64)
    }

    fn sgd() -> ModuleOptimizer {
        ModuleOptimizer::from(SgdConfig::new().init())
    }

    /// to_record / load_record must fully preserve a stateful optimizer's internal state (Adam's
    /// moment tensors and step counter) so that a step on the restored optimizer is numerically
    /// identical to one taken on the original.
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
                AdamConfig::new().init(),
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
            SgdConfig::new().init(),
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
            SgdConfig::new().init(),
            Some(GradientClipping::Value(threshold)),
        );

        // Large inputs produce gradients that clearly exceed the clipping threshold.
        let x = Tensor::<2>::random([2, 4], Distribution::Uniform(100.0, 200.0), &device);
        let weight_a_before = model.layer_a.weight.val();
        let weight_b_before = model.layer_b.weight.val();

        model = optim.step(
            LrPolicy::from(lr_value),
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

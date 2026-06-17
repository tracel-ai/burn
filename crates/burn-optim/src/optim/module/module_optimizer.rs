use burn_core as burn;

use super::Optimizer;
use crate::{
    DynOptimizer, DynState, LearningRate, MultiGradientsParams, OptimizerRecord, StateSink,
    StateSource, grad_clipping::GradientClipping, optim::GradientsParams, optim::state::join_path,
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

/// Optimizes a whole module by applying a per-parameter [`Optimizer`] to each of its parameters.
///
/// It is non-generic over the module and optimizer: any `O: Optimizer` is type-erased behind a
/// dynamic optimizer, and per-parameter states are kept as type-erased states keyed by
/// [`ParamId`](burn::module::ParamId). Build one with `optimizer.into()` or
/// `OptimizerConfig::init()`.
#[derive(Clone)]
pub struct ModuleOptimizer {
    optim: Arc<dyn DynOptimizer>,
    states: HashMap<ParamId, DynState>,
    grad_clipping: Option<GradientClipping>,
}

impl<O> From<O> for ModuleOptimizer
where
    O: Optimizer,
{
    fn from(optim: O) -> Self {
        Self {
            optim: Arc::new(optim),
            states: HashMap::new(),
            grad_clipping: None,
        }
    }
}

impl ModuleOptimizer {
    /// Check if the optimizer has gradient clipping.
    pub fn has_gradient_clipping(&self) -> bool {
        self.grad_clipping.is_some()
    }

    /// Access the gradient clipping.
    pub fn grad_clipping(&self) -> Option<&GradientClipping> {
        self.grad_clipping.as_ref()
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
        self.grad_clipping = Some(gradient_clipping);
        self
    }

    fn step_common<M: AutodiffModule>(
        &mut self,
        lr: LearningRate,
        module: M,
        mut grads: GradAdaptor,
    ) -> M {
        module.map(&mut ModuleOptimizerMapper::new(
            &self.optim,
            &mut self.states,
            &mut grads,
            lr,
            self.grad_clipping.as_ref(),
        ))
    }
}

impl ModuleOptimizer {
    /// Update the `module` parameters with the given `gradients`, advancing the optimizer state.
    pub fn step<M: AutodiffModule>(
        &mut self,
        lr: LearningRate,
        module: M,
        grads: GradientsParams,
    ) -> M {
        self.step_common(lr, module, grads.into())
    }

    /// Like [`step`](Self::step), but accumulating gradients sourced from multiple devices.
    pub fn step_multi<M: AutodiffModule>(
        &mut self,
        lr: LearningRate,
        module: M,
        grads: MultiGradientsParams,
    ) -> M {
        self.step_common(lr, module, grads.into())
    }

    /// Decompose the optimizer state into a serializable [`OptimizerRecord`].
    pub fn to_record(&self) -> OptimizerRecord {
        let mut tensors = Vec::new();
        let mut scalars = BTreeMap::new();

        for (id, state) in self.states.iter() {
            let prefix = id.val().to_string();
            let mut sink = StateSink::default();
            self.optim.state_flatten(&prefix, state, &mut sink);

            // Persist the parameter rank explicitly so the state can be reconstructed even when it
            // carries no tensors, and without inferring the rank from tensor shapes.
            scalars.insert(
                join_path(&prefix, RANK_KEY),
                burn_pack::Scalar::from(state.rank()),
            );

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

        OptimizerRecord { tensors, scalars }
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
            // Skip parameters whose state can't be reconstructed (truncated/foreign record); they
            // are re-initialized lazily on the next step rather than aborting the load.
            if let Some(state) = self
                .optim
                .state_unflatten(rank, &prefix, &mut source, &device)
            {
                states.insert(ParamId::from(id), state);
            }
        }

        self.states = states;
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
    optimizer: &'a Arc<dyn DynOptimizer>,
    states: &'a mut HashMap<ParamId, DynState>,
    grads: &'a mut GradAdaptor,
    lr: LearningRate,
    grad_clipping: Option<&'a GradientClipping>,
}

impl ModuleMapper for ModuleOptimizerMapper<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let grad = self.grads.remove(id);

        let tensor = if let Some((grad, device)) = grad {
            let is_require_grad = tensor.is_require_grad();
            #[cfg(feature = "std")]
            let is_distributed = tensor.is_distributed();

            let (key, state) = self.states.remove_entry(&id).unzip();
            let tensor = if tensor.device() != device {
                tensor.to_device(&device)
            } else {
                tensor
            };

            debug_assert_eq!(
                grad.device(),
                device,
                "The gradient is on the provided device"
            );
            let clipped_grad: Tensor<D> = if let Some(g_clipping) = self.grad_clipping {
                g_clipping.clip_gradient(grad)
            } else {
                grad
            };

            debug_assert_eq!(
                tensor.device(),
                device,
                "Tensor and gradients are on the same device."
            );

            let (tensor, state) = self.optimizer.step_dyn(
                D,
                self.lr,
                tensor.inner().into_bridge(),
                clipped_grad.into_bridge(),
                state.map(|state| self.optimizer.to_device_dyn(state, &device)),
            );

            if let Some(state) = state {
                self.states.insert(key.unwrap_or(id), state);
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

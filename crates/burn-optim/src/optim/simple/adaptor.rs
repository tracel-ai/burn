use burn_core as burn;

use super::OptimizerStep;
use crate::{
    DynOptimizer, DynState, LearningRate, MultiGradientsParams,
    grad_clipping::GradientClipping,
    optim::{GradientsParams, Optimizer},
};

use burn::module::{AutodiffModule, ModuleMapper, Param, ParamId};
use burn::tensor::{Device, Tensor};
use hashbrown::HashMap;
use std::sync::Arc;

/// Wrapper struct that adapts any [simple optimizer](SimpleOptimizer) into
/// an [optimizer](Optimizer).
#[derive(Clone)]
pub struct ModuleOptimizer {
    optim: Arc<dyn DynOptimizer>,
    states: HashMap<ParamId, DynState>,
    grad_clipping: Option<GradientClipping>,
}

impl<O> From<O> for ModuleOptimizer
where
    O: OptimizerStep,
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
        module.map(&mut SimpleOptimizerMapper::new(
            &self.optim,
            &mut self.states,
            &mut grads,
            lr,
            self.grad_clipping.as_ref(),
        ))
    }
}

impl ModuleOptimizer {
    pub fn step<M: AutodiffModule>(
        &mut self,
        lr: LearningRate,
        module: M,
        grads: GradientsParams,
    ) -> M {
        self.step_common(lr, module, grads.into())
    }

    pub fn step_multi<M: AutodiffModule>(
        &mut self,
        lr: LearningRate,
        module: M,
        grads: MultiGradientsParams,
    ) -> M {
        self.step_common(lr, module, grads.into())
    }

    // pub fn to_record(&self) -> Self::Record {
    //     todo!()
    // }

    // pub fn load_record(mut self, record: Self::Record) -> Self {
    //     // self.states = record;
    //     self
    // }
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
struct SimpleOptimizerMapper<'a> {
    optimizer: &'a Arc<dyn DynOptimizer>,
    states: &'a mut HashMap<ParamId, DynState>,
    grads: &'a mut GradAdaptor,
    lr: LearningRate,
    grad_clipping: Option<&'a GradientClipping>,
}

impl ModuleMapper for SimpleOptimizerMapper<'_> {
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

use burn_core as burn;
#[cfg(feature = "distributed")]
use burn_core::tensor::backend::distributed::DistributedParamId;

use super::{SimpleOptimizer, record::AdaptorRecord};
use crate::{
    LearningRate, MultiGradientsParams,
    grad_clipping::GradientClipping,
    optim::{GradientsParams, Optimizer},
};

use burn::module::{AutodiffModule, ModuleMapper, Param, ParamId};
use burn::tensor::{Device, Tensor};
use core::marker::PhantomData;
use hashbrown::HashMap;

/// Wrapper struct that adapts any [simple optimizer](SimpleOptimizer) into
/// an [optimizer](Optimizer).
#[derive(Clone)]
pub struct OptimizerAdaptor<O, M>
where
    O: SimpleOptimizer,
    M: AutodiffModule,
{
    optim: O,
    records: HashMap<ParamId, AdaptorRecord<O>>,
    module: PhantomData<M>,
    grad_clipping: Option<GradientClipping>,
}

impl<O, M> From<O> for OptimizerAdaptor<O, M>
where
    M: AutodiffModule,
    O: SimpleOptimizer,
{
    fn from(optim: O) -> Self {
        Self {
            optim,
            records: HashMap::new(),
            module: PhantomData,
            grad_clipping: None,
        }
    }
}

impl<O, M> OptimizerAdaptor<O, M>
where
    O: SimpleOptimizer,
    M: AutodiffModule,
{
    /// Access the wrapped [`SimpleOptimizer`].
    pub fn optim(&self) -> &O {
        &self.optim
    }

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

    fn step_common(&mut self, lr: LearningRate, module: M, mut grads: GradAdaptor) -> M {
        module.map(&mut SimpleOptimizerMapper::<O>::new(
            &self.optim,
            &mut self.records,
            &mut grads,
            lr,
            self.grad_clipping.as_ref(),
        ))
    }
}

impl<O, M> Optimizer<M> for OptimizerAdaptor<O, M>
where
    M: AutodiffModule,
    O: SimpleOptimizer,
{
    type Record = HashMap<ParamId, AdaptorRecord<O>>;

    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        self.step_common(lr, module, grads.into())
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, grads: MultiGradientsParams) -> M {
        self.step_common(lr, module, grads.into())
    }

    fn to_record(&self) -> Self::Record {
        self.records.clone()
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.records = record;
        self
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
struct SimpleOptimizerMapper<'a, O>
where
    O: SimpleOptimizer,
{
    optimizer: &'a O,
    records: &'a mut HashMap<ParamId, AdaptorRecord<O>>,
    grads: &'a mut GradAdaptor,
    lr: LearningRate,
    grad_clipping: Option<&'a GradientClipping>,
}

impl<O> ModuleMapper for SimpleOptimizerMapper<'_, O>
where
    O: SimpleOptimizer,
{
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let grad = self.grads.remove(id);

        let tensor = if let Some((grad, device)) = grad {
            let is_require_grad = tensor.is_require_grad();
            #[cfg(feature = "distributed")]
            let is_distributed = tensor.is_distributed();

            let (key, record) = self.records.remove_entry(&id).unzip();
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
            let clipped_grad = if let Some(g_clipping) = self.grad_clipping {
                g_clipping.clip_gradient(grad)
            } else {
                grad
            };

            debug_assert_eq!(
                tensor.device(),
                device,
                "Tensor and gradients are on the same device."
            );

            let (tensor, state) = self.optimizer.step(
                self.lr,
                tensor.inner(),
                clipped_grad,
                record.map(|record| O::to_device(record.into_state(), &device)),
            );

            if let Some(state) = state {
                self.records
                    .insert(key.unwrap_or(id), AdaptorRecord::from_state(state));
            }

            let mut tensor = Tensor::from_inner(tensor);
            if is_require_grad {
                tensor = tensor.require_grad();
            }
            #[cfg(feature = "distributed")]
            if is_distributed {
                tensor = tensor.set_distributed(DistributedParamId::from(id.val()))
            }

            tensor
        } else {
            tensor
        };

        Param::from_mapped_value(id, tensor, mapper)
    }
}

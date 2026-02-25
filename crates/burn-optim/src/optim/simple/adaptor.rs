use burn_core::{self as burn, prelude::Backend, tensor::Device};

use super::{SimpleOptimizer, record::AdaptorRecord};
use crate::{
    LearningRate, MultiGradientsParams,
    grad_clipping::GradientClipping,
    optim::{GradientsParams, Optimizer},
};

use burn::module::{AutodiffModule, ModuleMapper, Param, ParamId};
use burn::tensor::{Tensor, backend::AutodiffBackend};
use core::marker::PhantomData;
use hashbrown::HashMap;

/// Wrapper struct that adapts any [simple optimizer](SimpleOptimizer) into
/// an [optimizer](Optimizer).
#[derive(Clone)]
pub struct OptimizerAdaptor<O, M, B>
where
    O: SimpleOptimizer<B::InnerBackend>,
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    optim: O,
    records: HashMap<ParamId, AdaptorRecord<O, B>>,
    module: PhantomData<M>,
    grad_clipping: Option<GradientClipping>,
}

impl<O, B, M> From<O> for OptimizerAdaptor<O, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
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

impl<O, M, B> OptimizerAdaptor<O, M, B>
where
    O: SimpleOptimizer<B::InnerBackend>,
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
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

    #[cfg(test)]
    pub(crate) fn has_gradient_clipping(&self) -> bool {
        self.grad_clipping.is_some()
    }
}

impl<O, B, M> Optimizer<M, B> for OptimizerAdaptor<O, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
{
    type Record = HashMap<ParamId, AdaptorRecord<O, B>>;

    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        let mut grads = GradAdaptor::Single(grads);

        let mut mapper = SimpleOptimizerMapper::<M, B, O>::new(
            &self.optim,
            &mut self.records,
            &mut grads,
            lr,
            self.grad_clipping.as_ref(),
        );
        module.map(&mut mapper)
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, grads: crate::MultiGradientsParams) -> M {
        let mut grads = GradAdaptor::Multi(grads);

        let mut mapper = SimpleOptimizerMapper::<M, B, O>::new(
            &self.optim,
            &mut self.records,
            &mut grads,
            lr,
            self.grad_clipping.as_ref(),
        );
        module.map(&mut mapper)
    }

    fn to_record(&self) -> Self::Record {
        self.records.clone()
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.records = record;
        self
    }
}

enum GradAdaptor {
    Single(GradientsParams),
    Multi(MultiGradientsParams),
}

impl GradAdaptor {
    fn remove<B: Backend, const D: usize>(
        &mut self,
        id: ParamId,
    ) -> Option<(Tensor<B, D>, Device<B>)> {
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
struct SimpleOptimizerMapper<'a, M, B, O>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    optimizer: &'a O,
    records: &'a mut HashMap<ParamId, AdaptorRecord<O, B>>,
    grads: &'a mut GradAdaptor,
    lr: LearningRate,
    phantom: PhantomData<M>,
    grad_clipping: Option<&'a GradientClipping>,
}

impl<M, B, O> ModuleMapper<B> for SimpleOptimizerMapper<'_, M, B, O>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
    O: SimpleOptimizer<B::InnerBackend>,
{
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let grad = self.grads.remove(id);

        let tensor = if let Some((grad, device)) = grad {
            let is_require_grad = tensor.is_require_grad();
            let sharded_params = tensor.sharded_params();
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
            if let Some(params) = sharded_params {
                tensor = tensor.set_sharded_params(params.peer_id, params.op, params.param_id);
            }
            tensor
        } else {
            tensor
        };

        Param::from_mapped_value(id, tensor, mapper)
    }
}

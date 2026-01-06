use crate::renderer::MetricsRenderer;
use burn_core::module::AutodiffModule;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::{GradientsParams, MultiGradientsParams, Optimizer};

/// A training output.
pub struct TrainOutput<TO> {
    /// The gradients.
    pub grads: GradientsParams,

    /// The item.
    pub item: TO,
}

impl<TO> TrainOutput<TO> {
    /// Creates a new training output.
    ///
    /// # Arguments
    ///
    /// * `module` - The module.
    /// * `grads` - The gradients.
    /// * `item` - The item.
    ///
    /// # Returns
    ///
    /// A new training output.
    pub fn new<B: AutodiffBackend, M: AutodiffModule<B>>(
        module: &M,
        grads: B::Gradients,
        item: TO,
    ) -> Self {
        let grads = GradientsParams::from_grads(grads, module);
        Self { grads, item }
    }
}

/// Trait to be implemented for models to learn.
///
/// The [step](LearningStep::step) method needs to be manually implemented for all structs.
///
/// The [optimize](LearningStep::optimize) method can be overridden if you want to control how the
/// optimizer is used to update the model. This can be useful if you want to call custom mutable
/// functions on your model (e.g., clipping the weights) before or after the optimizer is used.
///
/// # Notes
///
/// To be used with the [Learner](crate::Learner) struct, the struct which implements this trait must
/// also implement the [AutodiffModule] trait, which is done automatically with the
/// [Module](burn_core::module::Module) derive.
pub trait LearningStep<TI, TO> {
    /// Runs a step for the learning, which executes the forward and backward passes.
    ///
    /// # Arguments
    ///
    /// * `item` - The input for the model.
    ///
    /// # Returns
    ///
    /// The output containing the model output and the gradients.
    fn step(&self, item: TI) -> TrainOutput<TO>;
    /// Optimize the current module with the provided gradients and learning rate.
    ///
    /// # Arguments
    ///
    /// * `optim`: Optimizer used for learning.
    /// * `lr`: The learning rate used for this step.
    /// * `grads`: The gradients of each parameter in the current model.
    ///
    /// # Returns
    ///
    /// The updated model.
    fn optimize<B, O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        B: AutodiffBackend,
        O: Optimizer<Self, B>,
        Self: AutodiffModule<B>,
    {
        optim.step(lr, self, grads)
    }
    /// Optimize the current module with the provided gradients and learning rate.
    ///
    /// # Arguments
    ///
    /// * `optim`: Optimizer used for learning.
    /// * `lr`: The learning rate used for this step.
    /// * `grads`: Multiple gradients associated to each parameter in the current model.
    ///
    /// # Returns
    ///
    /// The updated model.
    fn optimize_multi<B, O>(self, optim: &mut O, lr: f64, grads: MultiGradientsParams) -> Self
    where
        B: AutodiffBackend,
        O: Optimizer<Self, B>,
        Self: AutodiffModule<B>,
    {
        optim.step_multi(lr, self, grads)
    }
}

/// Trait to be implemented for validating models.
pub trait ValidStep<VI, VO> {
    /// Runs a validation step.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to validate on.
    ///
    /// # Returns
    ///
    /// The validation output.
    fn step(&self, item: VI) -> VO;
}

/// The result of a training, containing the model along with the [renderer](MetricsRenderer).
pub struct LearningResult<M> {
    /// The model with the learned weights.
    pub model: M,
    /// The renderer that can be used for follow up training and evaluation.
    pub renderer: Box<dyn MetricsRenderer>,
}

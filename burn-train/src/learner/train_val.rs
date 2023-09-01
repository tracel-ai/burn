use super::Learner;

use crate::{TrainEpoch, ValidEpoch};
use burn_core::data::dataloader::DataLoader;
use burn_core::lr_scheduler::LRScheduler;
use burn_core::module::ADModule;
use burn_core::optim::{GradientsParams, Optimizer};
use burn_core::tensor::backend::ADBackend;
use std::sync::Arc;

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
    pub fn new<B: ADBackend, M: ADModule<B>>(module: &M, grads: B::Gradients, item: TO) -> Self {
        let grads = GradientsParams::from_grads(grads, module);
        Self { grads, item }
    }
}

/// Trait to be implemented for training models.
///
/// The [step](TrainStep::step) method needs to be manually implemented for all structs.
///
/// The [optimize](TrainStep::optimize) method can be overridden if you want to control how the
/// optimizer is used to update the model. This can be useful if you want to call custom mutable
/// functions on your model (e.g., clipping the weights) before or after the optimizer is used.
///
/// # Notes
///
/// To be used with the [Learner](Learner) struct, the struct which implements this trait must
/// also implement the [ADModule](ADModule) trait, which is done automatically with the
/// [Module](burn_core::module::Module) derive.
pub trait TrainStep<TI, TO> {
    /// Runs the training step, which executes the forward and backward passes.
    ///
    /// # Arguments
    ///
    /// * `item` - The training input for the model.
    ///
    /// # Returns
    ///
    /// The training output containing the model output and the gradients.
    fn step(&self, item: TI) -> TrainOutput<TO>;
    /// Optimize the current module with the provided gradients and learning rate.
    ///
    /// # Arguments
    ///
    /// * `optim`: Optimizer used for training this model.
    /// * `lr`: The learning rate used for this step.
    /// * `grads`: The gradients of each parameter in the current model.
    ///
    /// # Returns
    ///
    /// The updated model.
    fn optimize<B, O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        B: ADBackend,
        O: Optimizer<Self, B>,
        Self: ADModule<B>,
    {
        optim.step(lr, self, grads)
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

impl<B, M, O, LR, TO, VO> Learner<B, M, O, LR, TO, VO>
where
    VO: Send + Sync + 'static,
    TO: Send + Sync + 'static,
    B: ADBackend,
    M: ADModule<B> + core::fmt::Display,
    O: Optimizer<M, B>,
    LR: LRScheduler,
{
    /// Fits the model.
    ///
    /// # Arguments
    ///
    /// * `dataloader_train` - The training dataloader.
    /// * `dataloader_valid` - The validation dataloader.
    ///
    /// # Returns
    ///
    /// The fitted model.
    pub fn fit<TI, VI>(
        mut self,
        dataloader_train: Arc<dyn DataLoader<TI>>,
        dataloader_valid: Arc<dyn DataLoader<VI>>,
    ) -> M
    where
        TI: Send + 'static,
        TO: Send + 'static,
        M: TrainStep<TI, TO> + Send + Clone + 'static,
        M::InnerModule: ValidStep<VI, VO>,
    {
        log::info!("Fitting {}", self.model.to_string());
        // The reference model is always on the first device provided.
        if let Some(device) = self.devices.get(0) {
            self.model = self.model.fork(device);
        }

        let starting_epoch = match self.checkpoint {
            Some(checkpoint) => {
                self = self.load_checkpoint(checkpoint);
                checkpoint
            }
            None => 1,
        };

        let mut model = self.model;
        let mut optim = self.optim;

        for epoch in starting_epoch..self.num_epochs + 1 {
            let epoch_train = TrainEpoch::new(
                dataloader_train.clone(),
                epoch,
                self.num_epochs,
                self.grad_accumulation,
            );

            if self.devices.len() > 1 {
                (model, optim) = epoch_train.run_multi_device(
                    model,
                    optim,
                    &mut self.lr_scheduler,
                    &mut self.callback,
                    self.devices.clone(),
                )
            } else {
                (model, optim) =
                    epoch_train.run(model, optim, &mut self.lr_scheduler, &mut self.callback);
            }

            let epoch_valid = ValidEpoch::new(dataloader_valid.clone(), epoch, self.num_epochs);
            epoch_valid.run(&model, &mut self.callback);

            Self::checkpoint(
                &model,
                &optim,
                &self.lr_scheduler,
                &self.checkpointer_model,
                &self.checkpointer_optimizer,
                &self.checkpointer_scheduler,
                epoch,
            );
        }

        model
    }
}

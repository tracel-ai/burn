use crate::components::LearnerComponents;
use crate::metric::processor::EventProcessor;
use crate::{Learner, TrainEpoch, ValidEpoch};
use burn_core::data::dataloader::DataLoader;
use burn_core::module::{AutodiffModule, Module};
use burn_core::optim::{GradientsParams, Optimizer};
use burn_core::tensor::backend::AutodiffBackend;
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
    pub fn new<B: AutodiffBackend, M: AutodiffModule<B>>(
        module: &M,
        grads: B::Gradients,
        item: TO,
    ) -> Self {
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
/// also implement the [AutodiffModule] trait, which is done automatically with the
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
        B: AutodiffBackend,
        O: Optimizer<Self, B>,
        Self: AutodiffModule<B>,
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

impl<LC: LearnerComponents> Learner<LC> {
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
    pub fn fit<InputTrain, InputValid, OutputTrain, OutputValid>(
        mut self,
        dataloader_train: Arc<dyn DataLoader<InputTrain>>,
        dataloader_valid: Arc<dyn DataLoader<InputValid>>,
    ) -> LC::Model
    where
        InputTrain: Send + 'static,
        InputValid: Send,
        OutputTrain: Send + 'static,
        OutputValid: Send,
        LC::Model: TrainStep<InputTrain, OutputTrain>,
        <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<InputValid, OutputValid>,
        LC::EventProcessor: EventProcessor<ItemTrain = OutputTrain, ItemValid = OutputValid>,
    {
        log::info!("Fitting the model:\n {}", self.model.to_string());
        // The reference model is always on the first device provided.
        if let Some(device) = self.devices.first() {
            self.model = self.model.fork(device);
        }

        let starting_epoch = match self.checkpoint {
            Some(checkpoint) => {
                if let Some(checkpointer) = &mut self.checkpointer {
                    (self.model, self.optim, self.lr_scheduler) = checkpointer.load_checkpoint(
                        self.model,
                        self.optim,
                        self.lr_scheduler,
                        &Default::default(), // Load the checkpoint on the default device.
                        checkpoint,
                    );
                }
                checkpoint + 1
            }
            None => 1,
        };

        for epoch in starting_epoch..self.num_epochs + 1 {
            let epoch_train = TrainEpoch::new(
                dataloader_train.clone(),
                epoch,
                self.num_epochs,
                self.grad_accumulation,
            );

            if self.devices.len() > 1 {
                (self.model, self.optim) = epoch_train.run_multi_device::<LC, OutputTrain>(
                    self.model,
                    self.optim,
                    &mut self.lr_scheduler,
                    &mut self.event_processor,
                    self.devices.clone(),
                    &self.interrupter,
                )
            } else {
                (self.model, self.optim) = epoch_train.run::<LC, OutputTrain>(
                    self.model,
                    self.optim,
                    &mut self.lr_scheduler,
                    &mut self.event_processor,
                    &self.interrupter,
                );
            }

            if self.interrupter.should_stop() {
                break;
            }

            let epoch_valid = ValidEpoch::new(dataloader_valid.clone(), epoch, self.num_epochs);
            epoch_valid.run::<LC, OutputValid>(
                &self.model,
                &mut self.event_processor,
                &self.interrupter,
            );

            if let Some(checkpointer) = &mut self.checkpointer {
                checkpointer.checkpoint(
                    &self.model,
                    &self.optim,
                    &self.lr_scheduler,
                    epoch,
                    &self.event_store,
                );
            }

            if let Some(early_stopping) = &mut self.early_stopping {
                if early_stopping.should_stop(epoch, &self.event_store) {
                    break;
                }
            }
        }

        // Display learner summary
        if let Some(summary) = self.summary {
            match summary.init() {
                Ok(summary) => {
                    // Drop event processor (includes renderer) so the summary is displayed
                    // when switching back to "main" screen
                    core::mem::drop(self.event_processor);
                    println!("{}", summary.with_model(self.model.to_string()))
                }
                Err(err) => log::error!("Could not retrieve learner summary:\n{err}"),
            }
        }

        self.model
    }
}

use super::Learner;
use crate::components_test::LearnerComponents;
use crate::metric_test::processor::{Event, EventProcessor, LearnerItem};
use burn_core::data::dataloader::DataLoader;
use burn_core::module::AutodiffModule;
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

impl<LC: LearnerComponents> Learner<LC> {
    /// Tests the model.
    ///
    /// # Arguments
    ///
    /// * `dataloader` - The testing dataloader.
    ///
    /// # Returns
    ///
    /// The tested model.
    pub fn test<InputTrain, OutputTrain>(
        mut self,
        dataloader: Arc<dyn DataLoader<InputTrain>>,
    ) -> LC::Model
    where
        InputTrain: Send + 'static,
        OutputTrain: Send + 'static,
        LC::Model: TrainStep<InputTrain, OutputTrain>,
    {
        let model = self.model;
        let mut processor = self.event_processor;

        log::info!("Executing testing");

        let mut iterator = dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            iteration += 1;
            log::info!("Iteration {}", iteration);

            let progress = iterator.progress();

            // TODO hmm
            let item = model.step(item);
            let item = LearnerItem::new(item.item, progress, 0, 0, iteration, None);

            processor.process(Event::ProcessedItem(item));
        }
        // processor.process_train(Event::EndEpoch(self.epoch));
        model
    }
}

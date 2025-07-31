use crate::components::{LearnerComponents, TrainBackend, ValidBackend};
use crate::metric::processor::{Event, EventProcessor};
use crate::{Learner, LearningStrategyExt};
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

pub(crate) type TrainLoader<LC, I> = Arc<dyn DataLoader<TrainBackend<LC>, I>>;
pub(crate) type ValidLoader<LC, I> = Arc<dyn DataLoader<ValidBackend<LC>, I>>;

/// Data loaders after having been prepared, split if needed
pub(crate) enum LearnerDataLoaders<LC: LearnerComponents, InputTrain, InputValid> {
    /// One dataloader for the training and one of the validation
    SingleTrainSingleValid {
        dataloader_train: TrainLoader<LC, InputTrain>,
        dataloader_valid: ValidLoader<LC, InputValid>,
    },
    /// Multiple data loaders for the training, one dataloader for the validation
    MultiTrainSingleValid {
        dataloader_train: Vec<TrainLoader<LC, InputTrain>>,
        dataloader_valid: ValidLoader<LC, InputValid>,
    },
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
    pub fn fit<TI, VI, TO, VO>(
        mut self,
        dataloader_train: Arc<dyn DataLoader<TrainBackend<LC>, TI>>,
        dataloader_valid: Arc<dyn DataLoader<ValidBackend<LC>, VI>>,
    ) -> LC::Model
    where
        TI: Send + 'static,
        VI: Send,
        TO: Send + 'static,
        VO: Send,
        LC::Model: TrainStep<TI, TO>,
        <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<VI, VO>,
        LC::EventProcessor: EventProcessor<ItemTrain = TO, ItemValid = VO>,
    {
        log::info!("Fitting the model:\n {}", self.model);

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

        let dataloaders = self
            .learning_strategy
            .prepare_dataloaders::<LC, TI, VI>(dataloader_train, dataloader_valid);

        self = self.learning_strategy.clone().prepare_model(self);

        self = self
            .learning_strategy
            .clone()
            .learn(self, dataloaders, starting_epoch);

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        self.event_processor.process_train(Event::End);

        self.display_learner_summary();

        self.model
    }

    fn display_learner_summary(&self) {
        if let Some(summary) = &self.summary {
            match summary.init() {
                Ok(summary) => {
                    println!("{}", summary.with_model(self.model.to_string()))
                }
                Err(err) => log::error!("Could not retrieve learner summary:\n{err}"),
            }
        }
    }
}

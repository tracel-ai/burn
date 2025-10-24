use crate::components::{
    InputTrain, InputValid, LearnerComponentTypes, LearningData, TrainBackend, ValidBackend,
};
#[cfg(feature = "ddp")]
use crate::ddp::DdpLearningStrategy;
use crate::multi::MultiDeviceLearningStrategy;
use crate::renderer::MetricsRenderer;
use crate::single::{CustomSingleDeviceLearningStrategy, SingleDeviceLearningStrategy};
use crate::{Learner, LearnerSummary, LearningMethod, LearningStrategy};
use burn_core::data::dataloader::DataLoader;
use burn_core::module::AutodiffModule;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::{GradientsParams, Optimizer};
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

/// A reference to the training split [DataLoader](DataLoader).
pub type TrainLoader<LC> = Arc<dyn DataLoader<TrainBackend<LC>, InputTrain<LC>>>;
/// A reference to the validation split [DataLoader](DataLoader).
pub type ValidLoader<LC> = Arc<dyn DataLoader<ValidBackend<LC>, InputValid<LC>>>;

/// The result of a training, containing the model along with the [renderer](MetricsRenderer).
pub struct TrainingResult<M> {
    /// The model trained.
    pub model: M,
    /// The renderer that can be used for follow up training and evaluation.
    pub renderer: Box<dyn MetricsRenderer>,
    /// A summary of the training.
    pub summary: Option<LearnerSummary>,
}

impl<LC: LearnerComponentTypes + Send + 'static> Learner<LC> {
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
    pub fn fit<T: TrainingLoop<LC>>(self, training: T) -> TrainingResult<LC::InnerModel> {
        training.run(self)
    }
}

pub struct Training<LC: LearnerComponentTypes> {
    training_loop: Box<dyn TrainingLoop<LC>>,
}

impl<LC: LearnerComponentTypes> Training<LC> {
    pub fn from_dataloaders<LM: LearningMethod>(
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
        learning_strategy: LM,
    ) -> Self {
        let training_loop = SupervisedLearning::new(dataloader_train, dataloader_valid, todo!());
        Self {
            training_loop: Box::new(training_loop),
        }
    }
}

#[derive(new)]
pub struct SupervisedLearning<LC: LearnerComponentTypes, LM: LearningMethod<LC>>
where
    LC::Model: TrainStep<
            <LC::LearningData as LearningData>::TrainInput,
            <LC::LearningData as LearningData>::TrainOutput,
        > + core::fmt::Display,
    LC::InnerModel: ValidStep<
            <LC::LearningData as LearningData>::ValidInput,
            <LC::LearningData as LearningData>::ValidOutput,
        >,
{
    dataloader_train: TrainLoader<LC>,
    dataloader_valid: ValidLoader<LC>,
    learning_strategy: LM,
}

pub trait SupervisedLearningTypes {
    type LC: LearnerComponentTypes<Model = Self::Model, InnerModel = Self::InnerModel>;
    type Model: TrainStep<
            <<Self::LC as LearnerComponentTypes>::LearningData as LearningData>::TrainInput,
            <<Self::LC as LearnerComponentTypes>::LearningData as LearningData>::TrainOutput,
        > + core::fmt::Display
        + AutodiffModule<<Self::LC as LearnerComponentTypes>::Backend, InnerModule = Self::InnerModel>
        + 'static;
    type InnerModel: ValidStep<
            <<Self::LC as LearnerComponentTypes>::LearningData as LearningData>::ValidInput,
            <<Self::LC as LearnerComponentTypes>::LearningData as LearningData>::ValidOutput,
        > + core::fmt::Display;
}

impl<LC: LearnerComponentTypes, LM: LearningMethod<LC>> TrainingLoop<LC>
    for SupervisedLearning<LC, LM>
where
    LC::Model: TrainStep<
            <LC::LearningData as LearningData>::TrainInput,
            <LC::LearningData as LearningData>::TrainOutput,
        > + core::fmt::Display,
    LC::InnerModel: ValidStep<
            <LC::LearningData as LearningData>::ValidInput,
            <LC::LearningData as LearningData>::ValidOutput,
        >,
{
    fn run(
        self: Box<Self>,
        learner: Learner<LC>,
    ) -> TrainingResult<<LC as LearnerComponentTypes>::InnerModel> {
        self.learning_strategy
            .fit(learner, self.dataloader_train, self.dataloader_valid)
    }
}

pub trait TrainingLoop<LC: LearnerComponentTypes> {
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
    fn run(self: Box<Self>, learner: Learner<LC>) -> TrainingResult<LC::InnerModel>;
}

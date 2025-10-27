use crate::components::{
    InputTrain, InputValid, LearnerComponentTypes, TrainBackend, ValidBackend,
};
#[cfg(feature = "ddp")]
use crate::ddp::DdpLearningStrategy;
use crate::multi::MultiDeviceLearningStrategy;
use crate::renderer::MetricsRenderer;
use crate::single::SingleDeviceLearningStrategy;
use crate::{Learner, LearningMethod, LearningStrategy};
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

pub(crate) type TrainLoader<LC> = Arc<dyn DataLoader<TrainBackend<LC>, InputTrain<LC>>>;
pub(crate) type ValidLoader<LC> = Arc<dyn DataLoader<ValidBackend<LC>, InputValid<LC>>>;

/// The result of a training, containing the model along with the [renderer](MetricsRenderer).
pub struct TrainingResult<M> {
    /// The model trained.
    pub model: M,
    /// The renderer that can be used for follow up training and evaluation.
    pub renderer: Box<dyn MetricsRenderer>,
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
    pub fn fit(
        self,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
    ) -> TrainingResult<LC::InnerModel> {
        log::info!("Fitting the model:\n {}", self.model);

        match &self.learning_strategy {
            LearningStrategy::SingleDevice(device) => {
                let single_device = SingleDeviceLearningStrategy::new(device.clone());
                single_device.fit(self, dataloader_train, dataloader_valid)
            }
            LearningStrategy::MultiDeviceNaive(devices) => {
                let multi_device = MultiDeviceLearningStrategy::new(devices.clone());
                multi_device.fit(self, dataloader_train, dataloader_valid)
            }

            #[cfg(feature = "ddp")]
            LearningStrategy::DistributedDataParallel { devices, config } => {
                let ddp = DdpLearningStrategy::new(devices.clone(), config.clone());
                ddp.fit(self, dataloader_train, dataloader_valid)
            }
        }
    }
}

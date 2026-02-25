use burn_core::data::dataloader::Progress;
use burn_core::module::AutodiffModule;
use burn_optim::GradientsAccumulator;
use std::sync::{Arc, Mutex};

use crate::SupervisedTrainingEventProcessor;
use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, TrainingItem};
use crate::{InferenceStep, Learner, LearningComponentsTypes, TrainLoader, ValidLoader};

/// A validation epoch.
#[derive(new)]
pub struct DdpValidEpoch<LC: LearningComponentsTypes> {
    dataloader: ValidLoader<LC>,
}

/// A training epoch.
#[derive(new)]
pub struct DdpTrainEpoch<LC: LearningComponentsTypes> {
    dataloader: TrainLoader<LC>,
    grad_accumulation: Option<usize>,
}

impl<LC: LearningComponentsTypes> DdpValidEpoch<LC> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        &self,
        model: &<LC as LearningComponentsTypes>::TrainingModel,
        global_progress: &Progress,
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing validation step for epoch {}", epoch);
        let model = model.valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = model.step(item);
            let item = TrainingItem::new(
                item,
                progress,
                global_progress.clone(),
                Some(iteration),
                None,
            );

            processor.process_valid(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        processor.process_valid(LearnerEvent::EndEpoch(epoch));
    }
}

impl<LC: LearningComponentsTypes> DdpTrainEpoch<LC> {
    /// Runs the training epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train.
    /// * `optim` - The optimizer to use.
    /// * `scheduler` - The learning rate scheduler to use.
    /// * `processor` - The event processor to use.
    ///
    /// # Returns
    ///
    /// The trained model and the optimizer.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        learner: &mut Learner<LC>,
        global_progress: &Progress,
        processor: Arc<Mutex<SupervisedTrainingEventProcessor<LC>>>,
        interrupter: &Interrupter,
        peer_count: usize,
        is_main: bool,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing training step for epoch {}", epoch,);

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        let mut grad_db = None;
        while let Some(item) = iterator.next() {
            for _ in 0..peer_count {
                iteration += 1;
                learner.lr_step();
            }
            log::info!("Iteration {iteration}");

            let mut progress = iterator.progress();
            progress.items_processed *= peer_count;
            progress.items_total *= peer_count;

            let item = learner.train_step(item);

            match self.grad_accumulation {
                Some(accumulation) => {
                    accumulator.accumulate(&learner.model(), item.grads);
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let grads = accumulator.grads();

                        learner.optimizer_step(grads);
                        accumulation_current = 0;
                    }
                }
                None => {
                    if let Some(last_grad) = grad_db {
                        learner.optimizer_step(last_grad);
                    }
                    grad_db = Some(item.grads);
                }
            }

            let item = TrainingItem::new(
                item.item,
                progress,
                global_progress.clone(),
                Some(iteration),
                Some(learner.lr_current()),
            );

            {
                let mut processor = processor.lock().unwrap();
                processor.process_train(LearnerEvent::ProcessedItem(item));
            }

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }

        if is_main {
            let mut processor = processor.lock().unwrap();
            processor.process_train(LearnerEvent::EndEpoch(epoch));
        }
    }
}

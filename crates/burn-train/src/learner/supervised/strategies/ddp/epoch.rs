use burn_core::data::dataloader::Progress;
use burn_optim::GradientsAccumulator;
use std::sync::{Arc, Mutex};

use crate::SupervisedTrainingEventProcessor;
use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, TrainingItem};
use crate::{InferenceStep, Learner, LearnerModel, TrainLoader, ValidLoader};

/// A validation epoch.
#[derive(new)]
pub struct DdpValidEpoch<M: LearnerModel> {
    dataloader: ValidLoader<M>,
}

/// A training epoch.
#[derive(new)]
pub struct DdpTrainEpoch<M: LearnerModel> {
    dataloader: TrainLoader<M>,
    grad_accumulation: Option<usize>,
}

impl<M: LearnerModel> DdpValidEpoch<M> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        &self,
        model: &M,
        global_progress: &Progress,
        processor: &mut SupervisedTrainingEventProcessor<M>,
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

            let item = InferenceStep::step(&model, item);
            let item = TrainingItem::new(item, progress, Some(iteration), None);

            processor.process_valid(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
    }
}

impl<M: LearnerModel> DdpTrainEpoch<M> {
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
    pub fn run(
        &self,
        learner: &mut Learner<M>,
        global_progress: &Progress,
        processor: Arc<Mutex<SupervisedTrainingEventProcessor<M>>>,
        interrupter: &Interrupter,
        peer_count: usize,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing training step for epoch {}", epoch,);

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

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
                    learner.optimizer_step(item.grads);
                }
            }

            let item = TrainingItem::new(
                item.item,
                progress,
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
    }
}

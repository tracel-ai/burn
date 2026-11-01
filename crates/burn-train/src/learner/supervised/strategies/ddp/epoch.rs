use burn_core::data::dataloader::Progress;
use burn_optim::GradientsAccumulator;
use std::sync::{Arc, Barrier, Mutex};

use super::step::StepCoordinator;
use super::strategy::DdpEventProcessor;
use crate::learner::base::Interrupter;
use crate::metric::processor::{LearnerEvent, TrainingItem};
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
    steps: Arc<StepCoordinator>,
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
        processor: &Mutex<DdpEventProcessor<M>>,
        interrupter: &Interrupter,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing validation step for epoch {}", epoch);
        let model = model.valid();

        let mut iterator = self.dataloader.iter();
        processor.lock().unwrap().valid(LearnerEvent::StartSplit {
            epoch_number: epoch,
            progress: iterator.progress(),
        });
        let mut iteration = 0;

        while !interrupter.should_stop() {
            let Some(item) = iterator.next() else { break };
            let item = match item {
                Ok(item) => item,
                Err(err) => {
                    interrupter.fail(err);
                    break;
                }
            };
            let progress = iterator.progress();
            iteration += 1;

            let item = InferenceStep::step(&model, item);
            let item = TrainingItem::new(item, progress, Some(iteration), None);

            processor
                .lock()
                .unwrap()
                .valid(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        processor.lock().unwrap().flush();

        let mut progress = iterator.progress();
        if !interrupter.should_stop() {
            progress.resolve_total();
        }
        processor
            .lock()
            .unwrap()
            .valid(LearnerEvent::EndSplit { epoch, progress });
        processor.lock().unwrap().flush();
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
        processor: Arc<Mutex<DdpEventProcessor<M>>>,
        interrupter: &Interrupter,
        worker_id: usize,
        barrier: &Barrier,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing training step for epoch {}", epoch,);

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        // Every worker must enter the same backward passes, even if a stop arrives after
        // the shared decision. The next rendezvous observes it before another backward.
        while let Some(item) = self.steps.next(&mut iterator, interrupter) {
            iteration += 1;
            log::info!("Iteration {iteration}");

            let progress = iterator.progress();

            let item = learner.train_step(item);

            match self.grad_accumulation {
                Some(accumulation) => {
                    accumulator.accumulate(&learner.model(), item.grads);
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let grads = accumulator.grads();

                        learner.lr_step();
                        learner.optimizer_step(grads);
                        accumulation_current = 0;
                    }
                }
                None => {
                    learner.lr_step();
                    learner.optimizer_step(item.grads);
                }
            }

            let item = TrainingItem::new(
                item.item,
                progress,
                Some(iteration),
                Some(learner.lr_current()),
            );
            processor.lock().unwrap().train(worker_id, item);
        }
        processor.lock().unwrap().flush();
        // All workers' metric failures must be visible before any final partial update.
        barrier.wait();
        if accumulation_current > 0 && !interrupter.should_stop() {
            let grads = accumulator.grads();
            learner.lr_step();
            learner.optimizer_step(grads);
        }
        let mut progress = iterator.progress();

        if !interrupter.should_stop() {
            progress.resolve_total();
        }
        processor.lock().unwrap().end_train(worker_id, progress);
    }
}

use crate::ddp::epoch::{DdpTrainEpoch, DdpValidEpoch};
use crate::ddp::strategy::WorkerComponents;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent};
use crate::single::TrainingLoop;
use crate::{
    Learner, LearnerModel, LearningCheckpointer, SupervisedTrainingEventProcessor, TrainLoader,
    ValidLoader,
};
use burn_core::tensor::Device;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A worker runs the model, syncing gradients using collective operations.
/// Event processing and validation is optional too.
pub(crate) struct DdpWorker<M: LearnerModel> {
    device: Device,
    learner: Learner<M>,
    event_processor: Arc<Mutex<SupervisedTrainingEventProcessor<M>>>,
    components: WorkerComponents,
    checkpointer: Option<LearningCheckpointer<M>>,
    dataloader_train: TrainLoader<M>,
    dataloader_valid: Option<ValidLoader<M>>,
    starting_epoch: usize,
    peer_count: usize,
    is_main: bool,
}

impl<M: LearnerModel> DdpWorker<M> {
    /// Starts a worker that runs the model in a data distributed parallel
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        device: Device,
        learner: Learner<M>,
        event_processor: Arc<Mutex<SupervisedTrainingEventProcessor<M>>>,
        components: WorkerComponents,
        checkpointer: Option<LearningCheckpointer<M>>,
        dataloader_train: TrainLoader<M>,
        dataloader_valid: Option<ValidLoader<M>>,
        starting_epoch: usize,
        peer_count: usize,
        is_main: bool,
    ) -> JoinHandle<M> {
        let worker = Self {
            device,
            learner,
            event_processor,
            components,
            checkpointer,
            dataloader_train,
            dataloader_valid,
            starting_epoch,
            peer_count,
            is_main,
        };

        std::thread::spawn(|| worker.fit())
    }

    /// Fits the model,
    pub fn fit(mut self) -> M {
        let num_epochs = self.components.num_epochs;
        let interrupter = self.components.interrupter;

        // Changed the train epoch to keep the dataloaders
        let epoch_train = DdpTrainEpoch::<M>::new(
            self.dataloader_train.clone(),
            self.components.grad_accumulation,
        );
        let epoch_valid = self
            .dataloader_valid
            .map(|dataloader| DdpValidEpoch::<M>::new(dataloader));
        self.learner.fork(&self.device);
        self.learner.grad_sharded();

        for training_progress in TrainingLoop::new(self.starting_epoch, num_epochs) {
            let epoch = training_progress.items_processed;

            if self.is_main {
                self.event_processor
                    .lock()
                    .unwrap()
                    .process_train(LearnerEvent::StartSplit(self.components.train_total_items));
            }

            epoch_train.run(
                &mut self.learner,
                &training_progress,
                self.event_processor.clone(),
                &interrupter,
                self.peer_count,
            );

            if self.is_main {
                self.event_processor
                    .lock()
                    .unwrap()
                    .process_train(LearnerEvent::EndSplit(epoch));
            }

            if interrupter.should_stop() {
                break;
            }

            // Validation
            if let Some(runner) = &epoch_valid {
                {
                    self.event_processor
                        .lock()
                        .unwrap()
                        .process_valid(LearnerEvent::StartSplit(self.components.valid_total_items));
                }
                let mut event_processor = self.event_processor.lock().unwrap();
                runner.run(
                    &self.learner.model(),
                    &training_progress,
                    &mut event_processor,
                    &interrupter,
                );
                event_processor.process_valid(LearnerEvent::EndSplit(epoch));
                event_processor.process_train(LearnerEvent::EndEpoch(epoch));
            }

            if let Some(checkpointer) = &mut self.checkpointer {
                checkpointer.checkpoint(&self.learner, epoch, &self.components.event_store);
            }

            if let Some(early_stopping) = &mut self.components.early_stopping
                && early_stopping.should_stop(epoch, &self.components.event_store)
            {
                break;
            }
        }

        self.learner.model()
    }
}

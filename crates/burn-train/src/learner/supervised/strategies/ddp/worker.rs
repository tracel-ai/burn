use crate::ddp::epoch::{DdpTrainEpoch, DdpValidEpoch};
use crate::ddp::strategy::WorkerComponents;
use crate::single::TrainingLoop;
use crate::{
    Learner, LearningCheckpointer, LearningComponentsTypes, SupervisedTrainingEventProcessor,
    TrainLoader, TrainingBackend, ValidLoader,
};
use burn_collective::{self, CollectiveConfig, PeerId};
use burn_core::tensor::Device;
use burn_core::tensor::backend::AutodiffBackend;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A worker runs the model, syncing gradients using collective operations.
/// Event processing and validation is optional too.
pub(crate) struct DdpWorker<LC>
where
    LC: LearningComponentsTypes + Send + 'static,
{
    peer_id: PeerId,
    device: Device<TrainingBackend<LC>>,
    learner: Learner<LC>,
    event_processor: Arc<Mutex<SupervisedTrainingEventProcessor<LC>>>,
    components: WorkerComponents,
    checkpointer: Option<LearningCheckpointer<LC>>,
    dataloader_train: TrainLoader<LC>,
    dataloader_valid: Option<ValidLoader<LC>>,
    collective_config: CollectiveConfig,
    starting_epoch: usize,
    peer_count: usize,
    is_main: bool,
}

impl<LC> DdpWorker<LC>
where
    LC: LearningComponentsTypes + Send + 'static,
{
    /// Starts a worker that runs the model in a data distributed parallel
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        peer_id: PeerId,
        device: Device<TrainingBackend<LC>>,
        learner: Learner<LC>,
        event_processor: Arc<Mutex<SupervisedTrainingEventProcessor<LC>>>,
        components: WorkerComponents,
        checkpointer: Option<LearningCheckpointer<LC>>,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: Option<ValidLoader<LC>>,
        collective_config: CollectiveConfig,
        starting_epoch: usize,
        peer_count: usize,
        is_main: bool,
    ) -> JoinHandle<<LC as LearningComponentsTypes>::TrainingModel> {
        let worker = Self {
            peer_id,
            device,
            learner,
            event_processor,
            components,
            checkpointer,
            dataloader_train,
            dataloader_valid,
            collective_config,
            starting_epoch,
            peer_count,
            is_main,
        };

        std::thread::spawn(|| worker.fit())
    }

    /// Fits the model,
    pub fn fit(mut self) -> <LC as LearningComponentsTypes>::TrainingModel {
        burn_collective::register::<<TrainingBackend<LC> as AutodiffBackend>::InnerBackend>(
            self.peer_id,
            self.device.clone(),
            self.collective_config.clone(),
        )
        .expect("Couldn't register for collective operations!");

        let num_epochs = self.components.num_epochs;
        let interrupter = self.components.interrupter;

        // Changed the train epoch to keep the dataloaders
        let epoch_train = DdpTrainEpoch::<LC>::new(
            self.dataloader_train.clone(),
            self.components.grad_accumulation,
        );
        let epoch_valid = self
            .dataloader_valid
            .map(|dataloader| DdpValidEpoch::<LC>::new(dataloader));
        self.learner.fork(&self.device);

        for training_progress in TrainingLoop::new(self.starting_epoch, num_epochs) {
            let epoch = training_progress.items_processed;

            epoch_train.run(
                &mut self.learner,
                &training_progress,
                self.event_processor.clone(),
                &interrupter,
                self.peer_id,
                self.peer_count,
                self.is_main,
            );

            if interrupter.should_stop() {
                break;
            }

            // Validation
            if let Some(runner) = &epoch_valid {
                let mut event_processor = self.event_processor.lock().unwrap();
                runner.run(
                    &self.learner.model(),
                    &training_progress,
                    &mut event_processor,
                    &interrupter,
                );
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

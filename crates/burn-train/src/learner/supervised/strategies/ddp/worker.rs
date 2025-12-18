use crate::ddp::epoch::{DdpTrainEpochV2, DdpValidEpochV2};
use crate::ddp::strategy::WorkerComponents;
use crate::{
    Learner, LearningCheckpointer, LearningComponentsTypes, ParadigmComponentsTypes,
    SupervisedLearningComponentsTypes, TrainBackend, TrainLoader, ValidLoader,
};
use burn_collective::{self, CollectiveConfig, PeerId};
use burn_core::tensor::Device;
use burn_core::tensor::backend::AutodiffBackend;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A worker runs the model, syncing gradients using collective operations.
/// Event processing and validation is optional too.
pub(crate) struct DdpWorkerV2<SC>
where
    SC: SupervisedLearningComponentsTypes + Send + 'static,
{
    peer_id: PeerId,
    device: Device<TrainBackend<SC::LC>>,
    learner: Learner<SC::LC>,
    event_processor: Arc<Mutex<<SC::PC as ParadigmComponentsTypes>::EventProcessor>>,
    components: WorkerComponents,
    checkpointer: Option<LearningCheckpointer<SC::LC, SC::PC>>,
    dataloader_train: TrainLoader<SC::LC, SC::LD>,
    dataloader_valid: Option<ValidLoader<SC::LC, SC::LD>>,
    collective_config: CollectiveConfig,
    starting_epoch: usize,
    peer_count: usize,
    is_main: bool,
}

impl<SC> DdpWorkerV2<SC>
where
    SC: SupervisedLearningComponentsTypes + Send + 'static,
{
    /// Starts a worker that runs the model in a data distributed parallel
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        peer_id: PeerId,
        device: Device<TrainBackend<SC::LC>>,
        learner: Learner<SC::LC>,
        event_processor: Arc<Mutex<<SC::PC as ParadigmComponentsTypes>::EventProcessor>>,
        components: WorkerComponents,
        checkpointer: Option<LearningCheckpointer<SC::LC, SC::PC>>,
        dataloader_train: TrainLoader<SC::LC, SC::LD>,
        dataloader_valid: Option<ValidLoader<SC::LC, SC::LD>>,
        collective_config: CollectiveConfig,
        starting_epoch: usize,
        peer_count: usize,
        is_main: bool,
    ) -> JoinHandle<<SC::LC as LearningComponentsTypes>::Model> {
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
    pub fn fit(mut self) -> <SC::LC as LearningComponentsTypes>::Model {
        burn_collective::register::<<TrainBackend<SC::LC> as AutodiffBackend>::InnerBackend>(
            self.peer_id,
            self.device.clone(),
            self.collective_config.clone(),
        )
        .expect("Couldn't register for collective operations!");

        let num_epochs = self.components.num_epochs;
        let interrupter = self.components.interrupter;

        // Changed the train epoch to keep the dataloaders
        let epoch_train = DdpTrainEpochV2::<SC>::new(
            self.dataloader_train.clone(),
            num_epochs,
            self.components.grad_accumulation,
        );
        let epoch_valid = self
            .dataloader_valid
            .map(|dataloader| DdpValidEpochV2::<SC>::new(dataloader, num_epochs));
        self.learner = self.learner.fork(&self.device);

        for epoch in self.starting_epoch..num_epochs + 1 {
            epoch_train.run(
                &mut self.learner,
                epoch,
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
                    &self.learner.model,
                    epoch,
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

        self.learner.model
    }
}

use crate::components::{LearnerComponents, TrainBackend, ValidBackend};
use crate::metric::processor::EventProcessor;
use crate::metric::store::EventStoreClient;
use crate::{
    EarlyStoppingStrategy, LearnerCheckpointer, TrainStep, TrainingInterrupter, ValidStep, ddp,
};
use burn_collective::{self, CollectiveConfig, PeerId};
use burn_core::data::dataloader::DataLoader;
use burn_core::module::AutodiffModule;
use burn_core::prelude::Backend;
use burn_core::tensor::backend::AutodiffBackend;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A worker runs the model, syncing gradients using collective operations.
/// Event processing and validation is optional too. 
pub(crate) struct DdpWorker<LC, InputTrain, OutputTrain, InputValid, OutputValid>
where
    LC: LearnerComponents,
    LC::Model: TrainStep<InputTrain, OutputTrain>,
    <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<InputValid, OutputValid>,
    LC::EventProcessor: EventProcessor<ItemTrain = OutputTrain, ItemValid = OutputValid>,
{
    pub peer_id: PeerId,
    pub device: <TrainBackend<LC> as Backend>::Device,
    pub model: LC::Model,
    pub optim: LC::Optimizer,
    pub early_stopping: Option<Arc<Mutex<dyn EarlyStoppingStrategy + Send>>>,
    pub event_processor: Option<LC::EventProcessor>,
    pub event_store: Arc<EventStoreClient>,
    pub checkpointer: Option<LearnerCheckpointer<LC>>,
    pub lr_scheduler: LC::LrScheduler,
    pub interrupter: TrainingInterrupter,
    pub dataloader_train: Arc<dyn DataLoader<TrainBackend<LC>, InputTrain>>,
    pub dataloader_valid: Option<Arc<dyn DataLoader<ValidBackend<LC>, InputValid>>>,
    pub collective_config: CollectiveConfig,
    pub starting_epoch: usize,
    pub num_epochs: usize,
    pub grad_accumulation: Option<usize>,
    _p: PhantomData<(OutputTrain, InputValid)>,
}

impl<LC, InputTrain, OutputTrain, InputValid, OutputValid>
    DdpWorker<LC, InputTrain, OutputTrain, InputValid, OutputValid>
where
    LC: LearnerComponents + 'static,
    InputTrain: Send + 'static,
    OutputTrain: Send + 'static,
    InputValid: Send + 'static,
    OutputValid: Send + 'static,
    LC::Model: TrainStep<InputTrain, OutputTrain>,
    <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<InputValid, OutputValid>,
    LC::EventProcessor: EventProcessor<ItemTrain = OutputTrain, ItemValid = OutputValid>,
{
    /// Starts a worker that runs the model in a data distributed parallel
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        peer_id: PeerId,
        device: <TrainBackend<LC> as Backend>::Device,
        model: LC::Model,
        optim: LC::Optimizer,
        early_stopping: Option<Arc<Mutex<dyn EarlyStoppingStrategy + Send>>>,
        event_processor: Option<LC::EventProcessor>,
        event_store: Arc<EventStoreClient>,
        checkpointer: Option<LearnerCheckpointer<LC>>,
        lr_scheduler: LC::LrScheduler,
        interrupter: TrainingInterrupter,
        dataloader_train: Arc<dyn DataLoader<TrainBackend<LC>, InputTrain> + Sync>,
        dataloader_valid: Option<Arc<dyn DataLoader<ValidBackend<LC>, InputValid>>>,
        collective_config: CollectiveConfig,
        starting_epoch: usize,
        num_epochs: usize,
        grad_accumulation: Option<usize>,
    ) -> JoinHandle<Self> {
        let worker = Self {
            peer_id,
            device,
            model,
            optim,
            early_stopping,
            event_processor,
            event_store,
            checkpointer,
            lr_scheduler,
            interrupter,
            dataloader_train,
            dataloader_valid,
            collective_config,
            starting_epoch,
            num_epochs,
            grad_accumulation,
            _p: PhantomData,
        };

        std::thread::spawn(|| worker.fit())
    }

    /// Fits the model,
    pub fn fit(mut self) -> Self {
        burn_collective::register::<<LC::Backend as AutodiffBackend>::InnerBackend>(
            self.peer_id,
            self.device.clone(),
            self.collective_config.clone(),
        )
        .expect("Couldn't register for collective operations!");

        // Changed the train epoch to keep the dataloaders
        let mut epoch_train = ddp::epoch::TrainEpoch::new(
            self.dataloader_train.clone(),
            self.starting_epoch,
            self.num_epochs,
            self.grad_accumulation,
        );

        for epoch in self.starting_epoch..self.num_epochs + 1 {
            (self.model, self.optim) = epoch_train.run::<LC, OutputTrain>(
                self.model,
                self.optim,
                &mut self.lr_scheduler,
                &mut self.event_processor,
                &self.interrupter,
                self.peer_id,
            );

            if self.interrupter.should_stop() {
                break;
            }

            // Validation
            if let Some(dataloader_valid) = &self.dataloader_valid {
                let epoch_valid =
                    ddp::epoch::ValidEpoch::new(dataloader_valid.clone(), epoch, self.num_epochs);
                epoch_valid.run::<LC, OutputValid>(
                    &self.model,
                    &mut self.event_processor,
                    &self.interrupter,
                );
            }

            if let Some(checkpointer) = &mut self.checkpointer {
                checkpointer.checkpoint(
                    &self.model,
                    &self.optim,
                    &self.lr_scheduler,
                    epoch,
                    &self.event_store,
                );
            }

            if let Some(early_stopping) = &mut self.early_stopping {
                let mut early_stopping = early_stopping.lock().unwrap();
                if early_stopping.should_stop(epoch, &self.event_store) {
                    break;
                }
            }
        }

        self
    }
}

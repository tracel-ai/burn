use crate::components::{LearnerComponentTypes, TrainBackend};
use crate::ddp::epoch::DdpValidEpoch;
use crate::learner::strategies::ddp;
use crate::metric::store::EventStoreClient;
use crate::{EarlyStoppingStrategyRef, Interrupter, LearnerCheckpointer, TrainLoader, ValidLoader};
use burn_collective::{self, CollectiveConfig, PeerId};
use burn_core::module::Module;
use burn_core::prelude::Backend;
use burn_core::tensor::backend::AutodiffBackend;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A worker runs the model, syncing gradients using collective operations.
/// Event processing and validation is optional too.
pub(crate) struct DdpWorker<LC>
where
    LC: LearnerComponentTypes + Send + 'static,
{
    peer_id: PeerId,
    device: <TrainBackend<LC> as Backend>::Device,
    model: LC::Model,
    optim: LC::Optimizer,
    early_stopping: Option<EarlyStoppingStrategyRef>,
    event_processor: Arc<Mutex<LC::EventProcessor>>,
    event_store: Arc<EventStoreClient>,
    checkpointer: Option<LearnerCheckpointer<LC>>,
    lr_scheduler: LC::LrScheduler,
    interrupter: Interrupter,
    dataloader_train: TrainLoader<LC>,
    dataloader_valid: Option<ValidLoader<LC>>,
    collective_config: CollectiveConfig,
    starting_epoch: usize,
    num_epochs: usize,
    grad_accumulation: Option<usize>,
    peer_count: usize,
    is_main: bool,
    _p: PhantomData<LC>,
}

impl<LC: LearnerComponentTypes> DdpWorker<LC>
where
    LC: LearnerComponentTypes + Send + 'static,
{
    /// Starts a worker that runs the model in a data distributed parallel
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        peer_id: PeerId,
        device: <TrainBackend<LC> as Backend>::Device,
        model: LC::Model,
        optim: LC::Optimizer,
        early_stopping: Option<EarlyStoppingStrategyRef>,
        event_processor: Arc<Mutex<LC::EventProcessor>>,
        event_store: Arc<EventStoreClient>,
        checkpointer: Option<LearnerCheckpointer<LC>>,
        lr_scheduler: LC::LrScheduler,
        interrupter: Interrupter,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: Option<ValidLoader<LC>>,
        collective_config: CollectiveConfig,
        starting_epoch: usize,
        num_epochs: usize,
        grad_accumulation: Option<usize>,
        peer_count: usize,
        is_main: bool,
    ) -> JoinHandle<LC::Model> {
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
            peer_count,
            is_main,
            _p: PhantomData,
        };

        std::thread::spawn(|| worker.fit())
    }

    /// Fits the model,
    pub fn fit(mut self) -> LC::Model {
        burn_collective::register::<<LC::Backend as AutodiffBackend>::InnerBackend>(
            self.peer_id,
            self.device.clone(),
            self.collective_config.clone(),
        )
        .expect("Couldn't register for collective operations!");

        // Changed the train epoch to keep the dataloaders
        let mut epoch_train = ddp::epoch::DdpTrainEpoch::<LC>::new(
            self.dataloader_train.clone(),
            self.starting_epoch,
            self.num_epochs,
            self.grad_accumulation,
        );
        self.model = self.model.fork(&self.device);

        for epoch in self.starting_epoch..self.num_epochs + 1 {
            (self.model, self.optim) = epoch_train.run(
                self.model,
                self.optim,
                &mut self.lr_scheduler,
                self.event_processor.clone(),
                &self.interrupter,
                self.peer_id,
                self.peer_count,
                self.is_main,
            );

            if self.interrupter.should_stop() {
                break;
            }

            // Validation
            if let Some(dataloader_valid) = &self.dataloader_valid {
                let epoch_valid =
                    DdpValidEpoch::<LC>::new(dataloader_valid.clone(), epoch, self.num_epochs);
                let mut event_processor = self.event_processor.lock().unwrap();
                epoch_valid.run(&self.model, &mut event_processor, &self.interrupter);
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

            if let Some(early_stopping) = &mut self.early_stopping
                && early_stopping.should_stop(epoch, &self.event_store)
            {
                break;
            }
        }

        self.model
    }
}

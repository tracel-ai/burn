use crate::components::{LearnerComponents, TrainBackend};
use crate::metric::processor::EventProcessor;
use crate::{TrainStep, TrainingInterrupter, ddp};
use burn_collective::{self, CollectiveConfig, PeerId};
use burn_core::data::dataloader::DataLoader;
use burn_core::tensor::backend::AutodiffBackend;
use std::marker::PhantomData;
use std::sync::Arc;

/// A Helper runs the model, but doesn't do a validation step, and doesn't process events.
pub(crate) struct DdpHelper<B, LC, InputTrain, OutputTrain>
where
    B: AutodiffBackend,
    LC: LearnerComponents<Backend = B> + 'static,
    InputTrain: Send + 'static,
    OutputTrain: Send + 'static,
{
    peer_id: PeerId,
    device: B::Device,
    model: LC::Model,
    optim: LC::Optimizer,
    lr_scheduler: LC::LrScheduler,
    interrupter: TrainingInterrupter,
    dataloader_train: Arc<dyn DataLoader<TrainBackend<LC>, InputTrain>>,
    collective_config: CollectiveConfig,
    starting_epoch: usize,
    num_epochs: usize,
    grad_accumulation: Option<usize>,
    _p: PhantomData<OutputTrain>,
}

impl<B, LC, InputTrain, OutputTrain> DdpHelper<B, LC, InputTrain, OutputTrain>
where
    B: AutodiffBackend,
    LC: LearnerComponents<Backend = B>,
    InputTrain: Send + 'static,
    OutputTrain: Send + 'static,
    LC::Model: TrainStep<InputTrain, OutputTrain>,
    LC::EventProcessor: EventProcessor<ItemTrain = OutputTrain>,
{
    /// Starts a worker that doesn't have an event processor and that doesn't do the validation step
    pub fn start_helper(
        peer_id: PeerId,
        device: B::Device,
        model: LC::Model,
        optim: LC::Optimizer,
        lr_scheduler: LC::LrScheduler,
        interrupter: TrainingInterrupter,
        dataloader_train: Arc<dyn DataLoader<TrainBackend<LC>, InputTrain> + Sync>,
        collective_config: CollectiveConfig,
        starting_epoch: usize,
        num_epochs: usize,
        grad_accumulation: Option<usize>,
    ) {
        let worker = Self {
            peer_id,
            device,
            model,
            optim,
            lr_scheduler,
            interrupter,
            dataloader_train,
            collective_config,
            starting_epoch,
            num_epochs,
            grad_accumulation,
            _p: PhantomData,
        };

        std::thread::spawn(|| worker.fit());
    }

    /// Fits the model,
    pub fn fit(mut self) -> LC::Model {
        burn_collective::register::<B::InnerBackend>(
            self.peer_id,
            self.device,
            self.collective_config,
        )
        .expect("Couldn't register for collective operations!");

        // Changed the train epoch to keep the dataloaders
        let mut epoch_train = ddp::epoch::TrainEpoch::new(
            self.dataloader_train,
            self.starting_epoch,
            self.num_epochs,
            self.grad_accumulation,
        );

        for _epoch in self.starting_epoch..self.num_epochs + 1 {
            (self.model, self.optim) = epoch_train.run::<LC, OutputTrain>(
                self.model,
                self.optim,
                &mut self.lr_scheduler,
                &mut None,
                &self.interrupter,
                self.peer_id,
            );

            if self.interrupter.should_stop() {
                break;
            }
        }

        self.model
    }
}

use crate::components::{LearnerComponents, TrainBackend, ValidBackend};
use crate::ddp::DdpLearner;
use crate::metric::processor::{Event, EventProcessor};
use crate::{TrainStep, ValidStep, ddp};
use burn_collective::{self, PeerId};
use burn_core::data::dataloader::DataLoader;
use burn_core::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use std::marker::PhantomData;
use std::sync::Arc;

/// Implements the [fit](Self::fit) function that the main device in a DDP runs. This includes
/// event processing as well as validation.
pub(crate) struct DdpMaster<B, LC, InputTrain, OutputTrain, InputValid, OutputValid>
where
    B: AutodiffBackend,
    LC: LearnerComponents<Backend = B> + 'static,
    InputTrain: Send + 'static,
    OutputTrain: Send + 'static,
    InputValid: Send + 'static,
    OutputValid: Send + 'static,
{
    _p: PhantomData<(B, LC, InputTrain, OutputTrain, InputValid, OutputValid)>,
}

impl<B, LC, InputTrain, OutputTrain, InputValid, OutputValid>
    DdpMaster<B, LC, InputTrain, OutputTrain, InputValid, OutputValid>
where
    B: AutodiffBackend,
    LC: LearnerComponents<Backend = B>,
    InputTrain: Send + 'static,
    OutputTrain: Send + 'static,
    InputValid: Send + 'static,
    OutputValid: Send + 'static,
    LC::Model: TrainStep<InputTrain, OutputTrain>,
    <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<InputValid, OutputValid>,
    LC::EventProcessor: EventProcessor<ItemTrain = OutputTrain, ItemValid = OutputValid>,
{
    /// Fits the model,
    pub fn fit(
        peer_id: PeerId,
        device: B::Device,
        mut learner: DdpLearner<LC>,
        starting_epoch: usize,
        dataloader_train: Arc<dyn DataLoader<TrainBackend<LC>, InputTrain>>,
        dataloader_valid: Arc<dyn DataLoader<ValidBackend<LC>, InputValid>>,
    ) -> DdpLearner<LC> {
        burn_collective::register::<B::InnerBackend>(
            peer_id,
            device,
            learner.collective_config.clone(),
        )
        .expect("Couldn't register for collective operations!");

        let mut inner = learner.inner;

        // Changed the train epoch to keep the dataloaders
        let mut epoch_train = ddp::epoch::TrainEpoch::new(
            dataloader_train,
            starting_epoch,
            inner.num_epochs,
            inner.grad_accumulation,
        );

        let mut event_processor = Some(inner.event_processor);
        for epoch in starting_epoch..inner.num_epochs + 1 {
            (inner.model, inner.optim) = epoch_train.run::<LC, OutputTrain>(
                inner.model,
                inner.optim,
                &mut inner.lr_scheduler,
                &mut event_processor,
                &inner.interrupter,
                peer_id,
            );

            if inner.interrupter.should_stop() {
                break;
            }

            // Validation
            let epoch_valid =
                ddp::epoch::ValidEpoch::new(dataloader_valid.clone(), epoch, inner.num_epochs);
            epoch_valid.run::<LC, OutputValid>(
                &inner.model,
                &mut event_processor,
                &inner.interrupter,
            );

            if let Some(checkpointer) = &mut inner.checkpointer {
                checkpointer.checkpoint(
                    &inner.model,
                    &inner.optim,
                    &inner.lr_scheduler,
                    epoch,
                    &inner.event_store,
                );
            }

            if let Some(early_stopping) = &mut inner.early_stopping {
                if early_stopping.should_stop(epoch, &inner.event_store) {
                    break;
                }
            }
        }

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        inner.event_processor = event_processor.unwrap();
        inner.event_processor.process_train(Event::End);

        // Display learner summary
        if let Some(summary) = &inner.summary {
            match summary.init() {
                Ok(summary) => {
                    println!("{}", summary.with_model(inner.model.to_string()))
                }
                Err(err) => log::error!("Could not retrieve learner summary:\n{err}"),
            }
        }

        learner.inner = inner;

        learner
    }
}

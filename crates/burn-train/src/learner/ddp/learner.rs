use crate::ddp::DdpHelper;
use crate::ddp::DdpMaster;
use burn_core::data::dataloader::split::split_dataloader;
use burn_core::module::Module;
use burn_core::{data::dataloader::DataLoader, module::AutodiffModule};
use std::sync::Arc;

use crate::metric::processor::Event;
use crate::{
    TrainStep, ValidStep,
    components::{LearnerComponents, TrainBackend, ValidBackend},
    ddp::DdpLearner,
    metric::processor::EventProcessor,
};

impl<LC: LearnerComponents + 'static> DdpLearner<LC> {
    /// Fits the model using collective operations in a Distributed Data Parallel.
    ///
    /// # Arguments
    ///
    /// * `dataloader_train` - The training dataloader.
    /// * `dataloader_valid` - The validation dataloader.
    ///
    /// # Returns
    ///
    /// The fitted model.
    pub fn fit<InputTrain, InputValid, OutputTrain, OutputValid>(
        mut self,
        mut dataloader_train: Arc<dyn DataLoader<TrainBackend<LC>, InputTrain>>,
        mut dataloader_valid: Arc<dyn DataLoader<ValidBackend<LC>, InputValid>>,
    ) -> LC::Model
    where
        InputTrain: Send + 'static,
        InputValid: Send + 'static,
        OutputTrain: Send + 'static,
        OutputValid: Send + 'static,
        LC::Model: TrainStep<InputTrain, OutputTrain>,
        <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<InputValid, OutputValid>,
        LC::EventProcessor: EventProcessor<ItemTrain = OutputTrain, ItemValid = OutputValid>,
    {
        log::info!("Fitting the model:\n {}", self.inner.model);

        if self.inner.devices.len() <= 1 {
            panic!("Should have more than one device")
        }

        // The reference model is always on the first device provided.
        let main_device = self.inner.devices[0].clone();

        self.inner.model = self.inner.model.fork(&main_device);
        dataloader_train = dataloader_train.to_device(&main_device);
        dataloader_valid = dataloader_valid.to_device(&main_device);

        let starting_epoch = match self.inner.checkpoint {
            Some(checkpoint) => {
                if let Some(checkpointer) = &mut self.inner.checkpointer {
                    (self.inner.model, self.inner.optim, self.inner.lr_scheduler) = checkpointer
                        .load_checkpoint(
                            self.inner.model,
                            self.inner.optim,
                            self.inner.lr_scheduler,
                            &main_device,
                            checkpoint,
                        );
                }
                checkpoint + 1
            }
            None => 1,
        };

        let mut dataloaders_train = split_dataloader(dataloader_train, &self.inner.devices);

        let collective_config = self.collective_config.clone();

        let lr_scheduler_clone = self.inner.lr_scheduler.clone();
        let interupter_clone = self.inner.interrupter.clone();

        // Spawn other workers for the other devices, starting with peer id 1
        let mut peer_id = 1;
        for device in self.inner.devices.drain(1..) {
            peer_id += 1;

            let model = self.inner.model.clone().fork(&device);
            let optim = self.inner.optim.clone();

            DdpHelper::<LC::Backend, LC, InputTrain, OutputTrain>::start_helper(
                peer_id.into(),
                device,
                model,
                optim,
                lr_scheduler_clone.clone(),
                interupter_clone.clone(),
                dataloaders_train.remove(0),
                collective_config.clone(),
                starting_epoch,
                self.inner.num_epochs,
                self.inner.grad_accumulation,
            );
        }

        // Run main device on this thread (does validation and has event processor), peer id = 0
        self = DdpMaster::fit(
            0.into(),
            main_device,
            self,
            starting_epoch,
            dataloaders_train.remove(0),
            dataloader_valid,
        );

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        self.inner.event_processor.process_train(Event::End);

        // Display learner summary
        if let Some(summary) = self.inner.summary {
            match summary.init() {
                Ok(summary) => {
                    println!("{}", summary.with_model(self.inner.model.to_string()))
                }
                Err(err) => log::error!("Could not retrieve learner summary:\n{err}"),
            }
        }

        self.inner.model
    }
}

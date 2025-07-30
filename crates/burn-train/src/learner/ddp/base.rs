use burn_collective::CollectiveConfig;

use crate::Learner;
use crate::ddp::DdpWorker;
use burn_core::data::dataloader::split::split_dataloader;
use burn_core::module::Module;
use burn_core::{data::dataloader::DataLoader, module::AutodiffModule};
use std::sync::Arc;

use crate::metric::processor::Event;
use crate::{
    TrainStep, ValidStep,
    components::{LearnerComponents, TrainBackend, ValidBackend},
    metric::processor::EventProcessor,
};

/// Wrapper for a learner that implements a Distributed Data Parallel, using collective operations
pub struct DdpLearner<LC: LearnerComponents> {
    pub(crate) inner: Learner<LC>,
    pub(crate) collective_config: CollectiveConfig,
}

impl<LC: LearnerComponents + 'static> DdpLearner<LC> {
    /// Constructs a new learner that uses DDP.
    /// 
    /// * `inner` - The learner to run on in a collective
    /// * `config` - Configurations for the collective operations
    pub fn new(inner: Learner<LC>, config: CollectiveConfig) -> Self {
        Self {
            inner,
            collective_config: config,
        }
    }

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

        // Split training data for each worker
        let mut dataloaders_train = split_dataloader(dataloader_train, &self.inner.devices);

        // Spawn other workers for the other devices, starting with peer id 1
        let mut peer_id = 1;
        let mut secondary_workers = vec![];
        for device in self.inner.devices.drain(1..) {
            peer_id += 1;

            let handle = DdpWorker::<LC, InputTrain, OutputTrain, InputValid, OutputValid>::start(
                peer_id.into(),
                device.clone(),
                self.inner.model.clone().fork(&device),
                self.inner.optim.clone(),
                self.inner.early_stopping.clone(),
                None,
                self.inner.event_store.clone(),
                None,
                self.inner.lr_scheduler.clone(),
                self.inner.interrupter.clone(),
                dataloaders_train.remove(0),
                None,
                self.collective_config.clone(),
                starting_epoch,
                self.inner.num_epochs,
                self.inner.grad_accumulation,
            );

            secondary_workers.push(handle);
        }

        // Start worker for main device
        // With validation data and event processor
        let main_handle =
            DdpWorker::<LC, InputTrain, OutputTrain, InputValid, OutputValid>::start(
                0.into(),
                main_device,
                self.inner.model,
                self.inner.optim,
                self.inner.early_stopping,
                Some(self.inner.event_processor),
                self.inner.event_store,
                self.inner.checkpointer,
                self.inner.lr_scheduler,
                self.inner.interrupter,
                dataloaders_train.remove(0),
                Some(dataloader_valid),
                self.collective_config,
                starting_epoch,
                self.inner.num_epochs,
                self.inner.grad_accumulation,
            );

        // Wait for all devices to finish
        for worker in secondary_workers {
            worker.join().expect("Distributed data parallel worker failed");
        }
        // Main worker had the event processor
        let main_worker = main_handle.join().expect("Distributed data parallel main worker failed");

        let mut event_processor = main_worker.event_processor.unwrap();
        let model = main_worker.model;

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        event_processor.process_train(Event::End);

        // Display learner summary
        if let Some(summary) = self.inner.summary {
            match summary.init() {
                Ok(summary) => {
                    println!("{}", summary.with_model(model.to_string()))
                }
                Err(err) => log::error!("Could not retrieve learner summary:\n{err}"),
            }
        }

        model
    }
}

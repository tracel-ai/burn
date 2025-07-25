use crate::{TrainOutput, TrainStep};
use burn_collective::{self, CollectiveConfig, DeviceId, SharedAllReduceParams, all_reduce};
use burn_core::data::dataloader::Progress;
use burn_core::module::{ModuleVisitor, ParamId};
use burn_core::optim::GradientsParams;
use burn_core::tensor::Tensor;
use burn_core::{
    data::dataloader::DataLoaderIterator, module::AutodiffModule, tensor::backend::AutodiffBackend,
};
use std::marker::PhantomData;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::spawn;


pub(crate) struct CollectiveWorker<B: AutodiffBackend, LC: LearnerComponents<Backend = B>> {
    device: B::Device,
    model: LC::Model,
    optim: LC::Optimizer,
    lr_scheduler: &mut LC::LrScheduler,
    event_processor: &mut LC::EventProcessor,
    interrupter: &TrainingInterrupter,
    dataloader_train: Arc<dyn DataLoader<TrainBackend<LC>, InputTrain>>,
    dataloader_valid: Arc<dyn DataLoader<ValidBackend<LC>, InputValid>>,
    collective_config: Arc<CollectiveConfig>,
    device_id: DeviceId,
}

impl<B, M, TI> CollectiveWorker<B, M, TI>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn start<TO>(
        device: B::Device,
        model: LC::Model,
        optim: LC::Optimizer,
        lr_scheduler: &mut LC::LrScheduler,
        event_processor: &mut LC::EventProcessor,
        interrupter: &TrainingInterrupter,
        dataloader_train: Arc<dyn DataLoader<TrainBackend<LC>, InputTrain>>,
        dataloader_valid: Arc<dyn DataLoader<ValidBackend<LC>, InputValid>>,
        collective_config: Arc<CollectiveConfig>,
        device_id: DeviceId,
    ) where
        TI: Send + 'static,
        TO: Send + 'static,
        M: TrainStep<TI, TO> + Send + 'static,
    {
        let worker = Self {
            device,
            model,
            optim,
            lr_scheduler,
            event_processor,
            interrupter,
            dataloader_train,
            dataloader_valid,
            collective_config,
            device_id,
        };

        spawn(worker.fit);
    }

    /// Fits the model,
    pub fn fit<InputTrain, InputValid, OutputTrain, OutputValid>(
        mut self,
    ) -> LC::Model
    where
        InputTrain: Send + 'static,
        InputValid: Send,
        OutputTrain: Send + 'static,
        OutputValid: Send,
        LC::Model: TrainStep<InputTrain, OutputTrain>,
        <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<InputValid, OutputValid>,
        LC::EventProcessor: EventProcessor<ItemTrain = OutputTrain, ItemValid = OutputValid>,
    {
        burn_collective::register::<B::InnerBackend>(device_id, num_devices, global_params)
            .expect("Couldn't register for collective operations!");

        // Changed the train epoch to keep the dataloaders
        let mut epoch_train = TrainEpoch::new(
            dataloaders_train,
            starting_epoch,
            self.num_epochs,
            self.grad_accumulation,
        );

        for epoch in starting_epoch..self.num_epochs + 1 {
            if self.devices.len() > 1 {
                (self.model, self.optim) = epoch_train.run_multi_device::<LC, OutputTrain>(
                    self.model,
                    self.optim,
                    &mut self.lr_scheduler,
                    &mut self.event_processor,
                    self.devices.clone(),
                    &self.interrupter,
                )
            } else {
                (self.model, self.optim) = epoch_train.run::<LC, OutputTrain>(
                    self.model,
                    self.optim,
                    &mut self.lr_scheduler,
                    &mut self.event_processor,
                    &self.interrupter,
                );
            }

            if self.interrupter.should_stop() {
                break;
            }

            // TODO: multi-device validation?
            let epoch_valid = ValidEpoch::new(dataloader_valid.clone(), epoch, self.num_epochs);
            epoch_valid.run::<LC, OutputValid>(
                &self.model,
                &mut self.event_processor,
                &self.interrupter,
            );

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
                if early_stopping.should_stop(epoch, &self.event_store) {
                    break;
                }
            }
        }

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        self.event_processor.process_train(Event::End);

        // Display learner summary
        if let Some(summary) = self.summary {
            match summary.init() {
                Ok(summary) => {
                    println!("{}", summary.with_model(self.model.to_string()))
                }
                Err(err) => log::error!("Could not retrieve learner summary:\n{err}"),
            }
        }

        self.model
    }

    /// 
    fn asdf<TO>(
        sender_output: Sender<TrainOutput<TO>>,
        receiver_input: Receiver<Message<M, TI>>,
        device: B::Device,
        device_id: DeviceId,
        all_reduce_params: SharedAllReduceParams,
    ) where
        TI: Send + 'static,
        TO: Send + 'static,
        M: TrainStep<TI, TO> + Send + 'static,
    {
        loop {
            match receiver_input.recv() {
                Ok(item) => {
                    let model = item.model.fork(&device);
                    let mut output = model.step(item.item);

                    // Sync with collective
                    output.grads =
                        output
                            .grads
                            .all_reduce(device_id, all_reduce_params.clone(), &model);

                    sender_output.send(output).unwrap();
                }
                Err(_err) => {
                    log::info!("Closing thread on device {device:?}");
                    break;
                }
            }
        }
    }
}

#[derive(new)]
struct GradientsParamsAllReduce<'a, M: AutodiffModule<B>, B: AutodiffBackend> {
    device_id: DeviceId,
    params: SharedAllReduceParams,
    grads: &'a mut GradientsParams,
    m: PhantomData<M>,
    b: PhantomData<B>,
}

impl<B, M> ModuleVisitor<B> for GradientsParamsAllReduce<'_, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
        let Some(mut grad) = self.grads.remove::<B::InnerBackend, D>(id) else {
            return;
        };

        grad = all_reduce::<B::InnerBackend, D>(self.device_id, grad, &self.params).unwrap();

        self.grads.register::<B::InnerBackend, D>(id, grad);
    }
}

trait GradientsParamsCollectiveExt {
    fn all_reduce<B: AutodiffBackend, M: AutodiffModule<B>>(
        self,
        device_id: burn_collective::DeviceId,
        params: burn_collective::SharedAllReduceParams,
        module: &M,
    ) -> Self;
}

impl GradientsParamsCollectiveExt for GradientsParams {
    /// All-Reduce the gradients for the given [module](AutodiffModule).
    fn all_reduce<B: AutodiffBackend, M: AutodiffModule<B>>(
        mut self,
        device_id: burn_collective::DeviceId,
        params: burn_collective::SharedAllReduceParams,
        module: &M,
    ) -> Self {
        let mut visitor = GradientsParamsAllReduce::<M, B>::new(device_id, params, &mut self);
        module.visit(&mut visitor);
        self
    }
}

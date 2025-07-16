use crate::{TrainOutput, TrainStep};
use burn_collective::config::CollectiveConfig;
use burn_collective::{GlobalRegisterParams, RegisterParams, SharedAllReduceParams, SharedRegisterParams};
use burn_core::collective;
use burn_core::data::dataloader::Progress;
use burn_core::{
    data::dataloader::DataLoaderIterator, module::AutodiffModule, tensor::backend::AutodiffBackend,
};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::spawn;

/// Multi devices train step using collective ops.
pub struct CollectiveTrainStep<B: AutodiffBackend, M, TI, TO> {
    workers: Vec<Worker<B, M, TI>>,
    receiver: Receiver<TrainOutput<TO>>,
}

struct Message<M, TI> {
    item: TI,
    model: M,
}

struct Worker<B: AutodiffBackend, M, TI> {
    sender_input: Sender<Message<M, TI>>,
    device: B::Device,
    device_id: burn_collective::DeviceId,
    register_shared_params: SharedRegisterParams,
    register_global_params: Option<GlobalRegisterParams>,
    all_reduce_params: SharedAllReduceParams,
}

impl<B, M, TI> Worker<B, M, TI>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn register(&self, item: TI, model: &M) {
        let message = Message {
            item,
            model: model.clone(),
        };
        self.sender_input.send(message).unwrap();
    }

    fn start<TO>(
        &self,
        sender_output: Sender<TrainOutput<TO>>,
        receiver_input: Receiver<Message<M, TI>>,
    ) where
        TI: Send + 'static,
        TO: Send + 'static,
        M: TrainStep<TI, TO> + Send + 'static,
    {
        let device = self.device.clone();

        let params = RegisterParams {
            device_id: self.device_id,
            shared: self.register_shared_params.clone(),
            global: self.register_global_params.clone(),
        };
        collective::register::<B::InnerBackend>(params)
            .expect("Couldn't register for collective operations!");

        let params = self.all_reduce_params.clone();
        let device_id = self.device_id;

        spawn(move || {
            loop {
                match receiver_input.recv() {
                    Ok(item) => {
                        let model = item.model.fork(&device);
                        let mut output = model.step(item.item);

                        // Sync with collective
                        output.grads = output.grads.all_reduce(device_id, params.clone(), &model);

                        sender_output.send(output).unwrap();
                    }
                    Err(_err) => {
                        log::info!("Closing thread on device {device:?}");
                        break;
                    }
                }
            }
        });
    }
}

impl<B, M, TI, TO> CollectiveTrainStep<B, M, TI, TO>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + TrainStep<TI, TO> + Send + Clone + 'static,
    TI: Send + 'static,
    TO: Send + 'static,
{
    /// Create a new multi devices train step.
    ///
    /// # Arguments
    ///
    /// * `devices` - Devices.
    ///
    /// # Returns
    ///
    /// CollectiveTrainStep instance.
    pub fn new(devices: &[B::Device], collective_config: &CollectiveConfig) -> Self
    where
        TI: Send + 'static,
    {
        let register_shared_params = collective_config.register_shared_params();
        let register_global_params = collective_config.register_global_params();
        let all_reduce_params = collective_config.all_reduce_params();
 
        let (sender_output, receiver_output) = std::sync::mpsc::channel();
        let workers = devices
            .iter()
            .enumerate()
            .map(|(idx, device)| {
                let (sender_input, receiver_input) = std::sync::mpsc::channel();
                let worker = Worker {
                    sender_input,
                    device: device.clone(),
                    device_id: (idx as u32).into(),
                    register_shared_params: register_shared_params.clone(),
                    register_global_params: register_global_params.clone(),
                    all_reduce_params: all_reduce_params.clone(),
                };

                worker.start(sender_output.clone(), receiver_input);
                worker
            })
            .collect();

        Self {
            workers,
            receiver: receiver_output,
        }
    }

    /// Collect outputs from workers for one step.
    ///
    /// # Arguments
    ///
    /// * `model` - Model.
    /// * `dataloaders` - The data loader for each worker.
    ///
    /// # Returns
    ///
    /// Outputs.
    pub fn step<'a>(
        &self,
        dataloaders: &mut [Box<dyn DataLoaderIterator<TI> + 'a>],
        model: &M,
    ) -> (TrainOutput<TO>, Progress) {
        let mut num_send = 0;

        let mut items_total = 0;
        let mut items_processed = 0;

        for (i, worker) in self.workers.iter().enumerate() {
            let dataloader = &mut dataloaders[i];
            if let Some(item) = dataloader.next() {
                worker.register(item, model);
                num_send += 1;
                let progress = dataloader.progress();
                items_total += progress.items_total;
                items_processed += progress.items_processed;
            }
        }

        // We only need to receive the output from one worker, since they have synced
        // the outputs should all be the same. But we will still wait for every worker to finish.
        let output = self.receiver.recv().unwrap();
        for _ in 1..num_send {
            self.receiver.recv().unwrap();
        }


        (output, Progress::new(items_processed, items_total))
    }
}

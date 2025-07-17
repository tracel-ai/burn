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
    collective_config: CollectiveConfig,
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
        device_id: DeviceId,
        num_devices: u32,
    ) where
        TI: Send + 'static,
        TO: Send + 'static,
        M: TrainStep<TI, TO> + Send + 'static,
    {
        let global_params = self.collective_config.register_global_params();
        let device = self.device.clone();
        let all_reduce_params = self.collective_config.all_reduce_params();

        spawn(move || {
            burn_collective::register::<B::InnerBackend>(device_id, num_devices, global_params)
                .expect("Couldn't register for collective operations!");

            Self::handle_input(
                sender_output,
                receiver_input,
                device,
                device_id,
                all_reduce_params,
            )
        });
    }

    fn handle_input<TO>(
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
        let (sender_output, receiver_output) = std::sync::mpsc::channel();
        let workers = devices
            .iter()
            .enumerate()
            .map(|(idx, device)| {
                let (sender_input, receiver_input) = std::sync::mpsc::channel();

                let collective_config = collective_config.clone();

                let worker = Worker {
                    sender_input,
                    device: device.clone(),
                    collective_config,
                };

                let device_id = (idx as u32).into();
                worker.start(
                    sender_output.clone(),
                    receiver_input,
                    device_id,
                    devices.len() as u32,
                );
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

use crate::components::{InputTrain, LearnerComponentTypes, OutputTrain};
use crate::{TrainOutput, TrainStep};
use burn_core::data::dataloader::DataLoaderIterator;
use burn_core::data::dataloader::Progress;
use burn_core::module::Module;
use burn_core::prelude::{Backend, DeviceOps};
use burn_core::tensor::backend::DeviceId;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::spawn;

/// Multi devices train step.
pub struct MultiDevicesTrainStep<LC: LearnerComponentTypes> {
    workers: Vec<Worker<LC>>,
    receiver: Receiver<MultiTrainOutput<OutputTrain<LC>>>,
}

struct Message<M, TI> {
    item: TI,
    model: M,
}

struct Worker<LC: LearnerComponentTypes> {
    sender_input: Sender<Message<LC::Model, InputTrain<LC>>>,
    device: <LC::Backend as Backend>::Device,
}

impl<LC: LearnerComponentTypes> Worker<LC> {
    fn register(&self, item: InputTrain<LC>, model: &LC::Model) {
        let message = Message {
            item,
            model: model.clone(),
        };
        self.sender_input.send(message).unwrap();
    }

    fn start(
        &self,
        sender_output: Sender<MultiTrainOutput<OutputTrain<LC>>>,
        receiver_input: Receiver<Message<LC::Model, InputTrain<LC>>>,
    ) {
        let device = self.device.clone();

        spawn(move || {
            loop {
                match receiver_input.recv() {
                    Ok(item) => {
                        // println!("On new step {:?}", device);
                        let model = item.model.fork(&device);
                        // println!("Model forked {:?}", device);
                        let output = model.step(item.item);
                        // println!("Model stepped {:?}", device);
                        let item = MultiTrainOutput {
                            output,
                            device: device.to_id(),
                        };

                        sender_output.send(item).unwrap();
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

/// Multiple output items.
pub struct MultiTrainOutput<TO> {
    /// The training output.
    pub output: TrainOutput<TO>,
    /// The device on which the computing happened.
    pub device: DeviceId,
}

impl<LC: LearnerComponentTypes> MultiDevicesTrainStep<LC> {
    /// Create a new multi devices train step.
    ///
    /// # Arguments
    ///
    /// * `devices` - Devices.
    ///
    /// # Returns
    ///
    /// MultiDevicesTrainStep instance.
    pub fn new(devices: &[<LC::Backend as Backend>::Device]) -> Self {
        let (sender_output, receiver_output) = std::sync::mpsc::channel();
        let workers = devices
            .iter()
            .map(|device| {
                let (sender_input, receiver_input) = std::sync::mpsc::channel();
                let worker = Worker {
                    sender_input,
                    device: device.clone(),
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
        dataloaders: &mut [Box<dyn DataLoaderIterator<InputTrain<LC>> + 'a>],
        model: &LC::Model,
    ) -> (Vec<MultiTrainOutput<OutputTrain<LC>>>, Progress) {
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

        let mut outputs = Vec::with_capacity(num_send);

        for _ in 0..num_send {
            let output = self.receiver.recv().unwrap();
            outputs.push(output);
        }

        (outputs, Progress::new(items_processed, items_total))
    }
}

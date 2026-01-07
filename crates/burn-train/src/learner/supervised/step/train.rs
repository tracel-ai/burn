use crate::{LearningComponentsTypes, TrainingModel};
use crate::{TrainOutput, TrainStep, TrainingBackend, TrainingModelInput, TrainingModelOutput};
use burn_core::data::dataloader::DataLoaderIterator;
use burn_core::data::dataloader::Progress;
use burn_core::module::Module;
use burn_core::prelude::DeviceOps;
use burn_core::tensor::Device;
use burn_core::tensor::backend::DeviceId;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::spawn;

/// Multi devices train step.
pub struct MultiDevicesTrainStep<LC: LearningComponentsTypes> {
    workers: Vec<Worker<LC>>,
    receiver: Receiver<MultiTrainOutput<TrainingModelOutput<LC>>>,
}

struct Message<M, TI> {
    item: TI,
    model: M,
}

struct Worker<LC: LearningComponentsTypes> {
    // Not that complex. Extracting into another type would only make it more confusing.
    #[allow(clippy::type_complexity)]
    sender_input: Sender<Message<TrainingModel<LC>, TrainingModelInput<LC>>>,
    device: Device<TrainingBackend<LC>>,
}

impl<LC: LearningComponentsTypes> Worker<LC> {
    fn register(&self, item: TrainingModelInput<LC>, model: &TrainingModel<LC>) {
        let message = Message {
            item,
            model: model.clone(),
        };
        self.sender_input.send(message).unwrap();
    }

    // Not that complex. Extracting into another type would only make it more confusing.
    #[allow(clippy::type_complexity)]
    fn start(
        &self,
        sender_output: Sender<MultiTrainOutput<TrainingModelOutput<LC>>>,
        receiver_input: Receiver<Message<TrainingModel<LC>, TrainingModelInput<LC>>>,
    ) {
        let device = self.device.clone();

        spawn(move || {
            loop {
                match receiver_input.recv() {
                    Ok(item) => {
                        let model = item.model.fork(&device);
                        let output = model.step(item.item);
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

impl<LC: LearningComponentsTypes> MultiDevicesTrainStep<LC> {
    /// Create a new multi devices train step.
    ///
    /// # Arguments
    ///
    /// * `devices` - Devices.
    ///
    /// # Returns
    ///
    /// MultiDevicesTrainStep instance.
    pub fn new(devices: &[Device<TrainingBackend<LC>>]) -> Self {
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
        dataloaders: &mut [Box<dyn DataLoaderIterator<TrainingModelInput<LC>> + 'a>],
        model: &TrainingModel<LC>,
    ) -> (Vec<MultiTrainOutput<TrainingModelOutput<LC>>>, Progress) {
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

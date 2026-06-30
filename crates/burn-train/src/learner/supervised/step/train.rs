use crate::LearnerModel;
use crate::{TrainOutput, TrainStep, TrainingModelInput, TrainingModelOutput};
use burn_core::data::dataloader::DataLoaderIterator;
use burn_core::data::dataloader::Progress;
use burn_core::tensor::Device;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::spawn;

/// Multi devices train step.
pub struct MultiDevicesTrainStep<M: LearnerModel> {
    workers: Vec<Worker<M>>,
    receiver: Receiver<MultiTrainOutput<TrainingModelOutput<M>>>,
}

struct Message<M, TI> {
    item: TI,
    model: M,
}

struct Worker<M: LearnerModel> {
    // Not that complex. Extracting into another type would only make it more confusing.
    // #[allow(clippy::type_complexity)]
    sender_input: Sender<Message<M, TrainingModelInput<M>>>,
    device: Device,
    device_id: usize,
}

impl<M: LearnerModel> Worker<M> {
    fn register(&self, item: TrainingModelInput<M>, model: &M) {
        let message = Message {
            item,
            model: model.clone(),
        };
        self.sender_input.send(message).unwrap();
    }

    // Not that complex. Extracting into another type would only make it more confusing.
    // #[allow(clippy::type_complexity)]
    fn start(
        &self,
        sender_output: Sender<MultiTrainOutput<TrainingModelOutput<M>>>,
        receiver_input: Receiver<Message<M, TrainingModelInput<M>>>,
    ) {
        let device = self.device.clone();
        let device_id = self.device_id;

        spawn(move || {
            loop {
                match receiver_input.recv() {
                    Ok(item) => {
                        let model = item.model.fork(&device);
                        let output = TrainStep::step(&model, item.item);
                        let item = MultiTrainOutput { output, device_id };

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
    /// The worker/device on which the computing happened.
    pub(crate) device_id: usize,
}

impl<M: LearnerModel> MultiDevicesTrainStep<M> {
    /// Create a new multi devices train step.
    ///
    /// # Arguments
    ///
    /// * `devices` - Devices.
    ///
    /// # Returns
    ///
    /// MultiDevicesTrainStep instance.
    pub fn new(devices: &[Device]) -> Self {
        let (sender_output, receiver_output) = std::sync::mpsc::channel();
        let workers = devices
            .iter()
            .enumerate()
            .map(|(device_id, device)| {
                let (sender_input, receiver_input) = std::sync::mpsc::channel();
                let worker = Worker {
                    sender_input,
                    device: device.clone(),
                    device_id,
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
        dataloaders: &mut [Box<dyn DataLoaderIterator<TrainingModelInput<M>> + 'a>],
        model: &M,
    ) -> (Vec<MultiTrainOutput<TrainingModelOutput<M>>>, Progress) {
        let mut num_send = 0;

        let mut items_total = 0;
        let mut items_processed = 0;
        let unit: Option<String> = Some("items".to_string());

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

        (outputs, Progress::new(items_processed, items_total, unit))
    }
}

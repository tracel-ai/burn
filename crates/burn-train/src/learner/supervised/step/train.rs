use crate::LearnerModel;
use crate::{TrainOutput, TrainStep, TrainingModelInput, TrainingModelOutput};
use burn_core::data::dataloader::DataLoaderIterator;
use burn_core::data::dataset::DatasetError;
use burn_core::data::dataloader::Progress;
use burn_core::tensor::Device;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::spawn;

/// Multi devices train step.
pub struct MultiDevicesTrainStep<M: LearnerModel> {
    workers: Vec<Worker<M>>,
    receiver: Receiver<WorkerMessage<TrainingModelOutput<M>>>,
}

struct Message<M, TI> {
    item: TI,
    model: M,
}

enum WorkerMessage<TO> {
    Output(MultiTrainOutput<TO>),
    Error(usize, String),
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic in training worker".to_string()
    }
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
        sender_output: Sender<WorkerMessage<TrainingModelOutput<M>>>,
        receiver_input: Receiver<Message<M, TrainingModelInput<M>>>,
    ) {
        let device = self.device.clone();
        let device_id = self.device_id;

        spawn(move || {
            loop {
                match receiver_input.recv() {
                    Ok(item) => {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let model = item.model.fork(&device);
                            TrainStep::step(&model, item.item)
                        }));

                        let message = match result {
                            Ok(output) => {
                                WorkerMessage::Output(MultiTrainOutput { output, device_id })
                            }
                            Err(payload) => {
                                WorkerMessage::Error(device_id, panic_message(payload.as_ref()))
                            }
                        };

                        if sender_output.send(message).is_err() {
                            break;
                        }
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
    ) -> Result<(Vec<MultiTrainOutput<TrainingModelOutput<M>>>, Progress), DatasetError> {
        let mut num_send = 0;

        let mut items_total = 0;
        let mut items_processed = 0;
        let unit: Option<String> = Some("items".to_string());

        for (i, worker) in self.workers.iter().enumerate() {
            let dataloader = &mut dataloaders[i];
            match dataloader.next() {
                Some(Ok(item)) => {
                    worker.register(item, model);
                    num_send += 1;
                    let progress = dataloader.progress();
                    items_total += progress.items_total;
                    items_processed += progress.items_processed;
                }
                Some(Err(err)) => return Err(err),
                None => {}
            }
        }

        let mut outputs = Vec::with_capacity(num_send);

        for _ in 0..num_send {
            match self.receiver.recv().unwrap() {
                WorkerMessage::Output(output) => outputs.push(output),
                WorkerMessage::Error(device_id, msg) => {
                    panic!("training worker on device {device_id} failed: {msg}");
                }
            }
        }

        Ok((outputs, Progress::new(items_processed, items_total, unit)))
    }
}

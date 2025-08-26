use burn_collective::{PeerId, ReduceOperation};
use burn_core::optim::GradientsParams;
use burn_core::tensor::backend::AutodiffBackend;
use burn_core::{lr_scheduler::LrScheduler, module::AutodiffModule, optim::GradientsAccumulator};
use std::marker::PhantomData;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use crate::metric::processor::{Event, EventProcessor, LearnerItem};
use crate::{TrainLoader, TrainStep, ValidLoader, ValidStep};
use crate::{components::LearnerComponentTypes, learner::base::TrainingInterrupter};

/// A validation epoch.
#[derive(new)]
pub struct DdpValidEpoch<LC: LearnerComponentTypes> {
    dataloader: ValidLoader<LC>,
    epoch: usize,
    epoch_total: usize,
}

/// A training epoch.
#[derive(new)]
pub struct DdpTrainEpoch<LC: LearnerComponentTypes> {
    dataloader: TrainLoader<LC>,
    epoch: usize,
    epoch_total: usize,
    grad_accumulation: Option<usize>,
}

impl<LC: LearnerComponentTypes> DdpValidEpoch<LC> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        &self,
        model: &LC::Model,
        processor: &mut LC::EventProcessor,
        interrupter: &TrainingInterrupter,
    ) {
        log::info!("Executing validation step for epoch {}", self.epoch);
        let model = model.valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = model.step(item);
            let item = LearnerItem::new(
                item,
                progress,
                self.epoch,
                self.epoch_total,
                iteration,
                None,
            );

            processor.process_valid(Event::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        processor.process_valid(Event::EndEpoch(self.epoch));
    }
}

impl<LC: LearnerComponentTypes> DdpTrainEpoch<LC> {
    /// Runs the training epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train.
    /// * `optim` - The optimizer to use.
    /// * `scheduler` - The learning rate scheduler to use.
    /// * `processor` - The event processor to use.
    ///
    /// # Returns
    ///
    /// The trained model and the optimizer.
    pub fn run(
        &mut self,
        mut model: LC::Model,
        mut optim: LC::Optimizer,
        scheduler: &mut LC::LrScheduler,
        processor: Arc<Mutex<LC::EventProcessor>>,
        interrupter: &TrainingInterrupter,
        peer_id: PeerId,
        peer_count: usize,
        is_main: bool,
    ) -> (LC::Model, LC::Optimizer) {
        log::info!("Executing training step for epoch {}", self.epoch,);

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        let grads_syncer = GradsSyncer::<LC::Backend, LC::Model>::new(true, peer_id);

        while let Some(item) = iterator.next() {
            let mut lr = 0.;
            for _ in 0..peer_count {
                iteration += 1;
                lr = scheduler.step();
            }
            log::info!("Iteration {iteration}");

            let mut progress = iterator.progress();
            progress.items_processed = progress.items_processed * peer_count;
            progress.items_total = progress.items_total * peer_count;

            let item = model.step(item);

            match self.grad_accumulation {
                Some(accumulation) => {
                    accumulator.accumulate(&model, item.grads);
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let grads = accumulator.grads();

                        // With double buffering, these are the previous iteration's gradients
                        let grads = grads_syncer.sync(grads);
                        if let Some(grads) = grads {
                            model = model.optimize(&mut optim, lr, grads);
                        }

                        accumulation_current = 0;
                    }
                }
                None => {
                    // With double buffering, these are the previous iteration's gradients
                    let grads = grads_syncer.sync(item.grads);

                    if let Some(grads) = grads {
                        model = model.optimize(&mut optim, lr, grads);
                    }
                }
            }

            let item = LearnerItem::new(
                item.item,
                progress,
                self.epoch,
                self.epoch_total,
                iteration,
                Some(lr),
            );

            {
                let mut processor = processor.lock().unwrap();
                processor.process_train(Event::ProcessedItem(item));
            }

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }

        if is_main {
            let mut processor = processor.lock().unwrap();
            processor.process_train(Event::EndEpoch(self.epoch));
        }

        self.epoch += 1;

        (model, optim)
    }
}

/// Worker that is responsible for syncing gradients for the DDP worker. With double buffering,
/// this allows for more optimization.
struct GradsSyncer<B: AutodiffBackend, M: AutodiffModule<B> + 'static> {
    msg_send: SyncSender<GradientsParams>,
    // Optional because with double buffering, the first iteration yields no gradients.
    result_recv: Receiver<Option<GradientsParams>>,

    _p: PhantomData<(B, M)>,
}

impl<B: AutodiffBackend, M: AutodiffModule<B> + 'static> GradsSyncer<B, M> {
    fn new(double_buffering: bool, peer_id: PeerId) -> Self {
        let (msg_send, msg_recv) = std::sync::mpsc::sync_channel::<GradientsParams>(1);
        let (result_send, result_recv) =
            std::sync::mpsc::sync_channel::<Option<GradientsParams>>(1);
        std::thread::spawn(move || {
            Self::run_worker(double_buffering, peer_id, result_send, msg_recv)
        });
        Self {
            msg_send,
            result_recv,
            _p: PhantomData,
        }
    }

    fn sync(&self, grads: GradientsParams) -> Option<GradientsParams> {
        self.msg_send.send(grads).unwrap();
        self.result_recv.recv().unwrap()
    }

    fn run_worker(
        double_buffering: bool,
        peer_id: PeerId,
        send: SyncSender<Option<GradientsParams>>,
        recv: Receiver<GradientsParams>,
    ) {
        let mut grads_buffer = None;

        while let Ok(new_grads) = recv.recv() {
            // Sync grads with collective
            let new_grads = new_grads
                .all_reduce::<B::InnerBackend>(peer_id, ReduceOperation::Sum)
                .expect("DDP worker could not sync gradients!");

            if double_buffering {
                let old_grads = grads_buffer.take();
                grads_buffer = Some(new_grads);

                send.send(old_grads).unwrap();
            } else {
                send.send(Some(new_grads)).unwrap();
            }
        }
        // GradsSyncer dropped, channel closed, this thread can end
    }
}

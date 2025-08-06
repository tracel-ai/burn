use burn_collective::all_reduce;
use burn_collective::{PeerId, ReduceOperation};
use burn_core::module::{ModuleVisitor, ParamId};
use burn_core::optim::GradientsParams;
use burn_core::tensor::Tensor;
use burn_core::tensor::backend::AutodiffBackend;
use burn_core::{lr_scheduler::LrScheduler, module::AutodiffModule, optim::GradientsAccumulator};
use std::marker::PhantomData;

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
        processor: &mut Option<LC::EventProcessor>,
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

            if let Some(processor) = processor {
                processor.process_valid(Event::ProcessedItem(item));
            }

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        if let Some(processor) = processor {
            processor.process_valid(Event::EndEpoch(self.epoch));
        }
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
        processor: &mut Option<LC::EventProcessor>,
        interrupter: &TrainingInterrupter,
        peer_id: PeerId,
    ) -> (LC::Model, LC::Optimizer) {
        log::info!("Executing training step for epoch {}", self.epoch,);

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
            iteration += 1;
            let lr = scheduler.step();
            log::info!("Iteration {iteration}");

            let progress = iterator.progress();
            let item = model.step(item);

            match self.grad_accumulation {
                Some(accumulation) => {
                    accumulator.accumulate(&model, item.grads);
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let mut grads = accumulator.grads();

                        // Sync grads with collective
                        grads = grads.all_reduce(peer_id, ReduceOperation::Mean, &model);

                        model = model.optimize(&mut optim, lr, grads);
                        accumulation_current = 0;
                    }
                }
                None => model = model.optimize(&mut optim, lr, item.grads),
            }

            let item = LearnerItem::new(
                item.item,
                progress,
                self.epoch,
                self.epoch_total,
                iteration,
                Some(lr),
            );

            if let Some(processor) = processor {
                processor.process_train(Event::ProcessedItem(item));
            }

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }

        if let Some(processor) = processor {
            processor.process_train(Event::EndEpoch(self.epoch));
        }

        self.epoch += 1;

        (model, optim)
    }
}

/// Params visitor for syncing gradients in a model, using [collective operations](burn_collective)
#[derive(new)]
struct GradientsParamsSync<'a, M: AutodiffModule<B>, B: AutodiffBackend> {
    peer_id: PeerId,
    op: burn_collective::ReduceOperation,
    grads: &'a mut GradientsParams,
    m: PhantomData<(M, B)>,
}

impl<B, M> ModuleVisitor<B> for GradientsParamsSync<'_, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
        let Some(mut grad) = self.grads.remove::<B::InnerBackend, D>(id) else {
            return;
        };

        grad = all_reduce::<B::InnerBackend, D>(self.peer_id, grad, self.op).unwrap();

        self.grads.register::<B::InnerBackend, D>(id, grad);
    }
}

trait GradientsParamsCollectiveExt {
    fn all_reduce<B: AutodiffBackend, M: AutodiffModule<B>>(
        self,
        device_id: burn_collective::PeerId,
        op: burn_collective::ReduceOperation,
        module: &M,
    ) -> Self;
}

impl GradientsParamsCollectiveExt for GradientsParams {
    /// All-Reduce the gradients for the given [module](AutodiffModule)
    /// using [collective operations](burn_collective)
    fn all_reduce<B: AutodiffBackend, M: AutodiffModule<B>>(
        mut self,
        device_id: burn_collective::PeerId,
        op: burn_collective::ReduceOperation,
        module: &M,
    ) -> Self {
        let mut visitor = GradientsParamsSync::<M, B>::new(device_id, op, &mut self);
        module.visit(&mut visitor);
        self
    }
}

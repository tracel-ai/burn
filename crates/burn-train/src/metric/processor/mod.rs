mod async_wrapper;
mod base;
mod full;
mod metrics;
mod minimal;
mod rl_metrics;
mod rl_processor;

pub use base::*;
pub(crate) use full::*;
pub(crate) use metrics::*;
pub(crate) use rl_metrics::*;
pub(crate) use rl_processor::*;

#[cfg(test)]
pub(crate) use minimal::*;

pub use async_wrapper::{AsyncProcessorEvaluation, AsyncProcessorTraining};

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::metric::{
        Adaptor, LossInput,
        processor::{EventProcessorTraining, LearnerEvent, MinimalEventProcessor, TrainingItem},
    };
    use burn_core::tensor::{ElementConversion, Tensor, backend::Backend};

    use super::ItemLazy;

    impl ItemLazy for f64 {
        type ItemSync = f64;

        fn sync(self) -> Self::ItemSync {
            self
        }
    }

    impl<B: Backend> Adaptor<LossInput<B>> for f64 {
        fn adapt(&self) -> LossInput<B> {
            let device = B::Device::default();
            LossInput::new(Tensor::from_data([self.elem::<B::FloatElem>()], &device))
        }
    }

    pub(crate) fn process_train(
        processor: &mut MinimalEventProcessor<f64, f64>,
        value: f64,
        epoch: usize,
    ) {
        let dummy_progress = burn_core::data::dataloader::Progress {
            items_processed: 1,
            items_total: 10,
        };
        let dummy_global_progress = burn_core::data::dataloader::Progress {
            items_processed: epoch,
            items_total: 3,
        };
        let dummy_iteration = Some(1);

        processor.process_train(LearnerEvent::ProcessedItem(TrainingItem::new(
            value,
            dummy_progress,
            dummy_global_progress,
            dummy_iteration,
            None,
        )));
    }

    pub(crate) fn end_epoch(processor: &mut MinimalEventProcessor<f64, f64>, epoch: usize) {
        processor.process_train(LearnerEvent::EndEpoch(epoch));
        processor.process_valid(LearnerEvent::EndEpoch(epoch));
    }
}

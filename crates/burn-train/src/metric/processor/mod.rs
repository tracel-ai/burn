mod async_wrapper;
mod base;
mod error;
mod full;
mod metrics;
mod minimal;
#[cfg(feature = "rl")]
mod rl_metrics;
#[cfg(feature = "rl")]
mod rl_processor;

pub use base::*;
pub use error::*;
pub(crate) use full::*;
pub(crate) use metrics::*;
#[cfg(feature = "rl")]
pub(crate) use rl_metrics::*;
#[cfg(feature = "rl")]
pub use rl_processor::*;

#[cfg(test)]
pub(crate) use minimal::*;

pub use async_wrapper::{AsyncProcessorEvaluation, AsyncProcessorTraining};

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::metric::{
        Adaptor, LossInput,
        processor::{EventProcessorTraining, LearnerEvent, MinimalEventProcessor, TrainingItem},
    };
    use burn_core::tensor::Tensor;

    use super::ItemLazy;

    impl ItemLazy for f64 {
        fn sync(self) -> Result<Self, burn_std::ExecutionError> {
            Ok(self)
        }
    }

    impl Adaptor<LossInput> for f64 {
        fn adapt(&self) -> LossInput {
            LossInput::new(Tensor::from_data([*self], &Default::default()))
        }
    }

    pub(crate) fn process_train(
        processor: &mut MinimalEventProcessor<f64, f64>,
        value: f64,
        epoch: usize,
    ) {
        let dummy_progress = burn_core::data::dataloader::Progress {
            items_processed: epoch,
            items_total: 3,
            unit: Some("items".to_string()),
        };
        let dummy_iteration = Some(1);

        processor
            .process_train(LearnerEvent::ProcessedItem(TrainingItem::new(
                value,
                dummy_progress,
                dummy_iteration,
                None,
            )))
            .unwrap();
    }

    pub(crate) fn start_epoch(
        processor: &mut MinimalEventProcessor<f64, f64>,
        epoch: usize,
        num_items: usize,
    ) {
        processor
            .process_train(LearnerEvent::StartSplit {
                epoch_number: epoch,
                total_items: num_items,
            })
            .unwrap();
        processor
            .process_valid(LearnerEvent::StartSplit {
                epoch_number: epoch,
                total_items: num_items,
            })
            .unwrap();
    }

    pub(crate) fn end_epoch(processor: &mut MinimalEventProcessor<f64, f64>, epoch: usize) {
        processor
            .process_train(LearnerEvent::EndSplit(epoch))
            .unwrap();
        processor
            .process_valid(LearnerEvent::EndSplit(epoch))
            .unwrap();
    }
}

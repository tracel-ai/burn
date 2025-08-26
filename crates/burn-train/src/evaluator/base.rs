use crate::{
    TrainingInterrupter,
    evaluator::components::{EvaluatorComponentTypes, TestStep},
    metric::{
        processor::{EvaluatorEvent, EventProcessorEvaluation, LearnerItem},
        store::EventStoreClient,
    },
    renderer::{EvaluationName, MetricsRenderer},
};
use burn_core::data::dataloader::DataLoader;
use std::sync::Arc;

/// Evaluates a model on a specific dataset.
pub struct Evaluator<EC: EvaluatorComponentTypes> {
    pub(crate) model: EC::Model,
    pub(crate) interrupter: TrainingInterrupter,
    pub(crate) event_processor: EC::EventProcessor,
    pub(crate) event_store: Arc<EventStoreClient>,
}

pub(crate) type TestBackend<EC> = <EC as EvaluatorComponentTypes>::Backend;
pub(crate) type TestInput<EC> = <EC as EvaluatorComponentTypes>::TestInput;

pub(crate) type TestLoader<EC> = Arc<dyn DataLoader<TestBackend<EC>, TestInput<EC>>>;

impl<EC: EvaluatorComponentTypes> Evaluator<EC> {
    /// TODO: Docs
    pub fn eval<S: core::fmt::Display>(
        mut self,
        name: S,
        dataloader: TestLoader<EC>,
    ) -> Box<dyn MetricsRenderer> {
        let name = EvaluationName::new(name);
        let mut iterator = dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = self.model.step(item);
            let item = LearnerItem::new(item, progress, 0, 1, iteration, None);

            self.event_processor
                .process_test(EvaluatorEvent::ProcessedItem(name.clone(), item));

            if self.interrupter.should_stop() {
                log::info!("Testing interrupted.");
                break;
            }
        }

        self.event_processor.renderer()
    }
}

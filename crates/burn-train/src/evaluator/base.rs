use crate::{
    AsyncProcessorEvaluation, FullEventProcessorEvaluation, InferenceStep, Interrupter,
    evaluator::components::EvaluatorComponentTypes,
    metric::processor::{EvaluatorEvent, EventProcessorEvaluation, LearnerItem},
    renderer::{EvaluationName, MetricsRenderer},
};
use burn_core::{data::dataloader::DataLoader, module::Module};
use std::sync::Arc;

pub(crate) type TestBackend<EC> = <EC as EvaluatorComponentTypes>::Backend;
pub(crate) type TestInput<EC> = <<EC as EvaluatorComponentTypes>::Model as InferenceStep>::Input;
pub(crate) type TestOutput<EC> = <<EC as EvaluatorComponentTypes>::Model as InferenceStep>::Output;

pub(crate) type TestLoader<EC> = Arc<dyn DataLoader<TestBackend<EC>, TestInput<EC>>>;

/// Evaluates a model on a specific dataset.
pub struct Evaluator<EC: EvaluatorComponentTypes> {
    pub(crate) model: EC::Model,
    pub(crate) interrupter: Interrupter,
    pub(crate) event_processor:
        AsyncProcessorEvaluation<FullEventProcessorEvaluation<TestOutput<EC>>>,
}

impl<EC: EvaluatorComponentTypes> Evaluator<EC> {
    /// Run the evaluation on the given dataset.
    ///
    /// The data will be stored and displayed under the provided name.
    pub fn eval<S: core::fmt::Display>(
        mut self,
        name: S,
        dataloader: TestLoader<EC>,
    ) -> Box<dyn MetricsRenderer> {
        // Move dataloader to the model device
        let dataloader = dataloader.to_device(self.model.devices().first().unwrap());

        let name = EvaluationName::new(name);
        let mut iterator = dataloader.iter();
        let mut iteration = 0;

        self.event_processor.process_test(EvaluatorEvent::Start);
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

use crate::{
    AsyncProcessorEvaluation, EvaluationItem, FullEventProcessorEvaluation, InferenceStep,
    Interrupter, LearnerSummaryConfig,
    evaluator::components::EvaluatorComponentTypes,
    metric::processor::{EvaluatorEvent, EventProcessorEvaluation},
    renderer::{EvaluationName, MetricsRenderer},
};
use burn_core::{data::dataloader::DataLoader, module::Module};
use std::sync::Arc;

pub(crate) type TestInput<EC> = <<EC as EvaluatorComponentTypes>::Model as InferenceStep>::Input;
pub(crate) type TestOutput<EC> = <<EC as EvaluatorComponentTypes>::Model as InferenceStep>::Output;

pub(crate) type TestLoader<EC> = Arc<dyn DataLoader<TestInput<EC>>>;

/// The result of an evaluation.
pub struct EvaluationResult {
    /// The renderer that can be used for follow up training and evaluation.
    pub renderer: Box<dyn MetricsRenderer>,
    /// The stop that ended evaluation early, if
    /// [`Interrupter::stop`](crate::Interrupter::stop) was called and no error happened.
    pub interrupted: Option<crate::Interruption>,
    /// The error that ended evaluation early, if it hit one.
    ///
    /// Both are `None` when every dataset was evaluated.
    pub error: Option<Arc<crate::TrainingError>>,
}

/// Evaluates a model on a specific dataset.
pub struct Evaluator<EC: EvaluatorComponentTypes> {
    pub(crate) model: EC::Model,
    pub(crate) interrupter: Interrupter,
    pub(crate) event_processor:
        AsyncProcessorEvaluation<FullEventProcessorEvaluation<TestOutput<EC>>>,
    /// Config for creating a summary of the evaluation
    pub summary: Option<LearnerSummaryConfig>,
}

impl<EC: EvaluatorComponentTypes> Evaluator<EC> {
    /// Run the evaluation on the given dataset.
    ///
    /// The data will be stored and displayed under the provided name.
    pub fn eval<S: core::fmt::Display>(
        self,
        name: S,
        dataloader: TestLoader<EC>,
    ) -> EvaluationResult {
        self.eval_all([(name, dataloader)])
    }

    /// Run the evaluation on multiple named datasets sequentially.
    ///
    /// Prefer this over calling [`eval`](Self::eval) in a loop — the progress logger
    /// receives the correct `total_tests` count and `end_test` is called between splits.
    pub fn eval_all<S: core::fmt::Display>(
        mut self,
        splits: impl IntoIterator<Item = (S, TestLoader<EC>)>,
    ) -> EvaluationResult {
        let splits: Vec<_> = splits.into_iter().collect();
        let total_tests = splits.len();

        self.interrupter.fail_on_error(
            self.event_processor
                .process_test(EvaluatorEvent::Start { total_tests }),
        );

        for (name, dataloader) in splits {
            let dataloader = dataloader.to_device(self.model.devices().first().unwrap());
            let name = EvaluationName::new(name);
            let total_items = dataloader.num_items();
            let mut iterator = dataloader.iter();
            let mut iteration = 0;

            self.interrupter.fail_on_error(
                self.event_processor
                    .process_test(EvaluatorEvent::StartTest(name.clone(), total_items)),
            );

            while let Some(item) = iterator.next() {
                let item = match item {
                    Ok(item) => item,
                    Err(err) => {
                        self.interrupter.fail(err);
                        break;
                    }
                };
                let progress = iterator.progress();
                iteration += 1;

                let item = self.model.step(item);
                let item = EvaluationItem::new(item, progress, Some(iteration));

                self.interrupter.fail_on_error(
                    self.event_processor
                        .process_test(EvaluatorEvent::ProcessedItem(name.clone(), item)),
                );

                if self.interrupter.should_stop() {
                    log::info!("Testing interrupted.");
                    break;
                }
            }

            self.interrupter
                .fail_on_error(self.event_processor.process_test(EvaluatorEvent::EndTest));

            // The remaining datasets are not evaluated once interrupted.
            if self.interrupter.should_stop() {
                break;
            }
        }

        let summary = self.summary.and_then(|summary| {
            summary
                .init()
                .map(|summary| summary.with_model(self.model.to_string()))
                .ok()
        });

        self.interrupter.fail_on_error(
            self.event_processor
                .process_test(EvaluatorEvent::End(summary)),
        );

        // Finish processing remaining events.
        self.interrupter.fail_on_error(self.event_processor.flush());

        EvaluationResult {
            renderer: self.event_processor.renderer(),
            interrupted: self.interrupter.interruption(),
            error: self.interrupter.error(),
        }
    }
}

use std::sync::Arc;

use crate::{
    EarlyStoppingStrategyRef, Interrupter, LearnerEvent, LearnerSummaryConfig,
    LearningCheckpointer, LearningResult, OffPolicyLearningComponentsTypes, RlEventProcessor,
    TrainingBackend,
    metric::{
        processor::EventProcessorTraining, rl_processor::RlEventProcessorTrain,
        store::EventStoreClient,
    },
};
use burn_rl::Agent;

/// Struct to minimise parameters passed to [SupervisedLearningStrategy::train].
/// These components are used during training.
pub struct RLComponents<OC: OffPolicyLearningComponentsTypes> {
    /// The total number of epochs
    pub num_epochs: usize,
    /// The epoch number from which to continue the training.
    pub checkpoint: Option<usize>,
    /// A checkpointer used to load and save learner checkpoints.
    // pub checkpointer: Option<LearningCheckpointer<OC::LC>>,
    pub checkpointer: Option<f64>,
    /// Enables gradients accumulation.
    pub grad_accumulation: Option<usize>,
    /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
    pub interrupter: Interrupter,
    /// Cloneable reference to an early stopping strategy.
    pub early_stopping: Option<EarlyStoppingStrategyRef>,
    /// An [EventProcessor](ParadigmComponentsTypes::EventProcessor) that processes events happening during training and validation.
    pub event_processor: RlEventProcessor<OC>,
    /// A reference to an [EventStoreClient](EventStoreClient).
    pub event_store: Arc<EventStoreClient>,
    /// Config for creating a summary of the learning
    pub summary: Option<LearnerSummaryConfig>,
}

/// Provides the `fit` function for any learning strategy
pub trait ReinforcementLearningStrategy<OC: OffPolicyLearningComponentsTypes> {
    /// Train the learner's model with this strategy.
    // fn train(
    //     &self,
    //     mut learner_agent: OC::LearningAgent,
    //     mut training_components: RLComponents<OC>,
    // ) -> LearningResult<<OC::LearningAgent as Agent<TrainingBackend<OC::LC>, OC::Env>>::Policy>
    fn train(
        &self,
        mut learner_agent: OC::LearningAgent,
        mut training_components: RLComponents<OC>,
    ) -> LearningResult<<OC::LearningAgent as Agent<OC::Backend, OC::Env>>::Policy> {
        let starting_epoch = 1;
        // let starting_epoch = match training_components.checkpoint {
        //     Some(checkpoint) => {
        //         if let Some(checkpointer) = &mut training_components.checkpointer {
        //             learner =
        //                 checkpointer.load_checkpoint(learner, &Default::default(), checkpoint);
        //         }
        //         checkpoint + 1
        //     }
        //     None => 1,
        // };

        let _summary_config = training_components.summary.clone();

        // Event processor start training
        training_components
            .event_processor
            .process_train(crate::metric::rl_processor::RlTrainingEvent::Start);

        // Training loop
        let (model, mut event_processor) =
            self.fit(training_components, &mut learner_agent, starting_epoch);

        // let summary = summary_config.and_then(|summary| {
        //     summary
        //         .init()
        //         .map(|summary| summary.with_model(model.to_string()))
        //         .ok()
        // });

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        // event_processor.process_train(LearnerEvent::End(summary));
        event_processor.process_train(crate::metric::rl_processor::RlTrainingEvent::End(None));

        // let model = model.valid();
        let renderer = event_processor.renderer();

        LearningResult { model, renderer }
    }

    /// Training loop for this strategy
    fn fit(
        &self,
        training_components: RLComponents<OC>,
        learner_agent: &mut OC::LearningAgent,
        starting_epoch: usize,
    ) -> (
        // <OC::LearningAgent as Agent<TrainingBackend<OC::LC>, OC::Env>>::Policy,
        <OC::LearningAgent as Agent<OC::Backend, OC::Env>>::Policy,
        RlEventProcessor<OC>,
    );
}

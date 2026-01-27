use std::sync::Arc;

use burn_core::prelude::Backend;

use crate::{
    Interrupter, LearnerSummaryConfig, OffPolicyConfig, RLCheckpointer, RLEvent,
    RLEventProcessorType, RLResult, ReinforcementLearningComponentsTypes,
    metric::{processor::EventProcessorTraining, store::EventStoreClient},
};

/// Struct to minimise parameters passed to [ReinforcementLearningStrategy::train].
pub struct RLComponents<RLC: ReinforcementLearningComponentsTypes> {
    /// The total number of environment steps.
    pub num_steps: usize,
    /// The step number from which to continue the training.
    pub checkpoint: Option<usize>,
    /// A checkpointer used to load and save learning checkpoints.
    pub checkpointer: Option<RLCheckpointer<RLC>>,
    /// Enables gradients accumulation.
    pub grad_accumulation: Option<usize>,
    /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
    pub interrupter: Interrupter,
    /// A [RLEventProcessor](crate::RLEventProcessor) that processes events happening during training and evaluation.
    pub event_processor: RLEventProcessorType<RLC>,
    /// A reference to an [EventStoreClient](EventStoreClient).
    pub event_store: Arc<EventStoreClient>,
    /// Config for creating a summary of the learning
    pub summary: Option<LearnerSummaryConfig>,
}

/// The strategy for reinforcement learning.
#[derive(Clone)]
pub enum RLStrategy<RLC: ReinforcementLearningComponentsTypes> {
    /// Training on one device
    OffPolicyStrategy((<RLC::Backend as Backend>::Device, OffPolicyConfig)),
    /// Training using a custom learning strategy
    Custom(CustomRLStrategy<RLC>),
}

/// A reference to an implementation of [ReinforcementLearningStrategy].
pub type CustomRLStrategy<LC> = Arc<dyn ReinforcementLearningStrategy<LC>>;

/// Provides the `fit` function for any learning strategy
pub trait ReinforcementLearningStrategy<RLC: ReinforcementLearningComponentsTypes> {
    /// Train the learner's model with this strategy.
    fn train(
        &self,
        mut learner_agent: RLC::LearningAgent,
        mut training_components: RLComponents<RLC>,
    ) -> RLResult<RLC::Policy> {
        let starting_epoch = match training_components.checkpoint {
            Some(checkpoint) => {
                if let Some(checkpointer) = &mut training_components.checkpointer {
                    learner_agent = checkpointer.load_checkpoint(
                        learner_agent,
                        &Default::default(),
                        checkpoint,
                    );
                }
                checkpoint + 1
            }
            None => 1,
        };

        let summary_config = training_components.summary.clone();

        // Event processor start training
        training_components
            .event_processor
            .process_train(RLEvent::Start);

        // Training loop
        let (policy, mut event_processor) =
            self.fit(training_components, &mut learner_agent, starting_epoch);

        let summary = summary_config.and_then(|summary| summary.init().ok());

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        // TODO: summary makes sense for RL?
        event_processor.process_train(RLEvent::End(summary));

        // let model = model.valid();
        let renderer = event_processor.renderer();

        RLResult { policy, renderer }
    }

    /// Training loop for this strategy
    fn fit(
        &self,
        training_components: RLComponents<RLC>,
        learner_agent: &mut RLC::LearningAgent,
        starting_epoch: usize,
    ) -> (RLC::Policy, RLEventProcessorType<RLC>);
}

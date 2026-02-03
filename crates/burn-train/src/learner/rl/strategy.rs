use std::sync::Arc;

use crate::{
    Interrupter, LearnerSummaryConfig, OffPolicyConfig, RLCheckpointer, RLComponentsTypes, RLEvent,
    RLEventProcessorType, RLResult,
    metric::{processor::EventProcessorTraining, store::EventStoreClient},
};

/// Struct to minimise parameters passed to [RLStrategy::train].
pub struct RLComponents<RLC: RLComponentsTypes> {
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
    /// An [EventProcessor](crate::EventProcessorTraining) that processes events happening during training and evaluation.
    pub event_processor: RLEventProcessorType<RLC>,
    /// A reference to an [EventStoreClient](EventStoreClient).
    pub event_store: Arc<EventStoreClient>,
    /// Config for creating a summary of the learning
    pub summary: Option<LearnerSummaryConfig>,
}

/// The strategy for reinforcement learning.
#[derive(Clone)]
pub enum RLStrategies<RLC: RLComponentsTypes> {
    /// Training on one device
    OffPolicyStrategy(OffPolicyConfig),
    /// Training using a custom learning strategy
    Custom(CustomRLStrategy<RLC>),
}

/// A reference to an implementation of [RLStrategy].
pub type CustomRLStrategy<LC> = Arc<dyn RLStrategy<LC>>;

/// Provides the `fit` function for any learning strategy
pub trait RLStrategy<RLC: RLComponentsTypes> {
    /// Train the learner agent with this strategy.
    fn train(
        &self,
        mut learner_agent: RLC::LearningAgent,
        mut training_components: RLComponents<RLC>,
        env_init: RLC::EnvInit,
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
        let (policy, mut event_processor) = self.learn(
            training_components,
            &mut learner_agent,
            starting_epoch,
            env_init,
        );

        let summary = summary_config.and_then(|summary| summary.init().ok());

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        // TODO: summary makes sense for RL?
        event_processor.process_train(RLEvent::End(summary));

        // let model = model.valid();
        let renderer = event_processor.renderer();

        RLResult { policy, renderer }
    }

    /// Training loop for this strategy
    fn learn(
        &self,
        training_components: RLComponents<RLC>,
        learner_agent: &mut RLC::LearningAgent,
        starting_epoch: usize,
        env_init: RLC::EnvInit,
    ) -> (RLC::Policy, RLEventProcessorType<RLC>);
}

use std::marker::PhantomData;

use burn_core::tensor::backend::AutodiffBackend;
use burn_rl::{Agent, Environment, LearnerAgent};

use crate::{AgentEvaluationEvent, AsyncProcessorTraining, ItemLazy, RLEvent};

/// All components used by the reinforcement learning paradigm, grouped in one trait.
pub trait ReinforcementLearningComponentsTypes {
    /// The backend used for training.
    type Backend: AutodiffBackend;
    /// The learning environement.
    type Env: Environment + 'static;
    /// The learning agent.
    type LearningAgent: LearnerAgent<
            Self::Backend,
            Self::Env,
            TrainingOutput = Self::TrainingOutput,
            DecisionContext = Self::ActionContext,
        > + Send
        + 'static;
    /// Additional data as context for an agent's action.
    type ActionContext: ItemLazy + Clone + Send + 'static;
    /// The output data of a training step.
    type TrainingOutput: ItemLazy + Clone + Send;
}

/// Concrete type that implements the [ReinforcementLearningComponentsTypes](ReinforcementLearningComponentsTypes) trait.
pub struct ReinforcementLearningComponentsMarker<B, E, A, TO, AC> {
    _backend: PhantomData<B>,
    _env: PhantomData<E>,
    _agent: PhantomData<A>,
    _training_output: PhantomData<TO>,
    _action_context: PhantomData<AC>,
}

impl<B, E, A, TO, AC> ReinforcementLearningComponentsTypes
    for ReinforcementLearningComponentsMarker<B, E, A, TO, AC>
where
    B: AutodiffBackend,
    E: Environment + 'static,
    A: LearnerAgent<B, E, TrainingOutput = TO, DecisionContext = AC> + Send + 'static,
    TO: ItemLazy + Clone + Send,
    AC: ItemLazy + Clone + Send + 'static,
{
    type Backend = B;
    type Env = E;
    type LearningAgent = A;
    type ActionContext = AC;
    type TrainingOutput = TO;
}

pub(crate) type RlPolicy<OC> =
    <<OC as ReinforcementLearningComponentsTypes>::LearningAgent as Agent<
        <OC as ReinforcementLearningComponentsTypes>::Backend,
        <OC as ReinforcementLearningComponentsTypes>::Env,
    >>::Policy;
pub(crate) type RlState<OC> =
    <<OC as ReinforcementLearningComponentsTypes>::Env as Environment>::State;
/// The event processor type for reinforcement learning.
pub type RLEventProcessorType<OC> = AsyncProcessorTraining<
    RLEvent<
        <OC as ReinforcementLearningComponentsTypes>::TrainingOutput,
        <OC as ReinforcementLearningComponentsTypes>::ActionContext,
    >,
    AgentEvaluationEvent<<OC as ReinforcementLearningComponentsTypes>::ActionContext>,
>;

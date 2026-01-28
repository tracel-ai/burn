use std::marker::PhantomData;

use burn_core::tensor::backend::AutodiffBackend;
use burn_rl::{AgentLearner, Environment, EnvironmentInit, Policy, PolicyState};

use crate::{AgentEvaluationEvent, AsyncProcessorTraining, ItemLazy, RLEvent};

/// All components used by the reinforcement learning paradigm, grouped in one trait.
pub trait RLComponentsTypes {
    /// The backend used for training.
    type Backend: AutodiffBackend;
    /// The learning environement.
    type Env: Environment<State = Self::State, Action = Self::Action> + 'static;
    /// Specifies how to initialize the environment.
    type EnvInit: EnvironmentInit<Self::Env> + Send + 'static;
    /// The learning agent.
    /// // TODO: type shit is weird here.
    type LearningAgent: AgentLearner<
            Self::Backend,
            TrainingOutput = Self::TrainingOutput,
            InnerPolicy = Self::Policy,
        > + Send
        + 'static;
    /// The policy used to take actions in the environment.
    type Policy: Policy<Self::Backend, ActionContext = Self::ActionContext> + Send + 'static;
    /// Additional data as context for an agent's action.
    type ActionContext: ItemLazy + Clone + Send + 'static;
    /// The output data of a training step.
    type TrainingOutput: ItemLazy + Clone + Send;
    /// The type of the environment state.
    type State: Into<<Self::Policy as Policy<Self::Backend>>::Input> + Clone;
    /// The type of the environment action.
    type Action: From<<Self::Policy as Policy<Self::Backend>>::Action>;
}

/// Concrete type that implements the [ReinforcementLearningComponentsTypes](ReinforcementLearningComponentsTypes) trait.
pub struct RLComponentsMarker<B, E, EI, A, TO, AC> {
    _backend: PhantomData<B>,
    _env: PhantomData<E>,
    _env_init: PhantomData<EI>,
    _agent: PhantomData<A>,
    _training_output: PhantomData<TO>,
    _action_context: PhantomData<AC>,
}

impl<B, E, EI, A, TO, AC> RLComponentsTypes for RLComponentsMarker<B, E, EI, A, TO, AC>
where
    B: AutodiffBackend,
    E: Environment + 'static,
    EI: EnvironmentInit<E> + Send + 'static,
    A: AgentLearner<B, TrainingOutput = TO> + Send + 'static,
    A::InnerPolicy: Policy<B, ActionContext = AC> + Send,
    TO: ItemLazy + Clone + Send,
    AC: ItemLazy + Clone + Send + 'static,
    E::State: Into<<A::InnerPolicy as Policy<B>>::Input> + Clone,
    E::Action: From<<A::InnerPolicy as Policy<B>>::Action>,
{
    type Backend = B;
    type Env = E;
    type EnvInit = EI;
    type LearningAgent = A;
    type Policy = A::InnerPolicy;
    type ActionContext = AC;
    type TrainingOutput = TO;
    type State = E::State;
    type Action = E::Action;
}

pub(crate) type RlPolicy<RLC> = <<RLC as RLComponentsTypes>::LearningAgent as AgentLearner<
    <RLC as RLComponentsTypes>::Backend,
>>::InnerPolicy;
/// The event processor type for reinforcement learning.
pub type RLEventProcessorType<RLC> = AsyncProcessorTraining<
    RLEvent<<RLC as RLComponentsTypes>::TrainingOutput, <RLC as RLComponentsTypes>::ActionContext>,
    AgentEvaluationEvent<<RLC as RLComponentsTypes>::ActionContext>,
>;
/// The record of the policy.
pub type RLPolicyRecord<RLC> = <<<RLC as RLComponentsTypes>::Policy as Policy<
    <RLC as RLComponentsTypes>::Backend,
>>::PolicyState as PolicyState<<RLC as RLComponentsTypes>::Backend>>::Record;
/// The record of the learning agent.
pub type RLAgentRecord<RLC> = <<RLC as RLComponentsTypes>::LearningAgent as AgentLearner<
    <RLC as RLComponentsTypes>::Backend,
>>::Record;

use std::marker::PhantomData;

use burn_core::tensor::backend::AutodiffBackend;
use burn_rl::{Batchable, Environment, EnvironmentInit, Policy, PolicyLearner, PolicyState};

use crate::{AgentEvaluationEvent, AsyncProcessorTraining, ItemLazy, RLEvent};

/// All components used by the reinforcement learning paradigm, grouped in one trait.
pub trait RLComponentsTypes {
    /// The backend used for training.
    type Backend: AutodiffBackend;
    /// The learning environement.
    type Env: Environment<State = Self::State, Action = Self::Action> + 'static;
    /// Specifies how to initialize the environment.
    type EnvInit: EnvironmentInit<Self::Env> + Send + 'static;
    /// The type of the environment state.
    type State: Into<<Self::Policy as Policy<Self::Backend>>::Observation> + Clone + Send + 'static;
    /// The type of the environment action.
    type Action: From<<Self::Policy as Policy<Self::Backend>>::Action>
        + Into<<Self::Policy as Policy<Self::Backend>>::Action>
        + Clone
        + Send
        + 'static;

    /// The policy used to take actions in the environment.
    type Policy: Policy<
            Self::Backend,
            Observation = Self::PolicyObs,
            ActionDistribution = Self::PolicyAD,
            Action = Self::PolicyAction,
            ActionContext = Self::ActionContext,
            PolicyState = Self::PolicyState,
        > + Send
        + 'static;
    /// The policy's observation type.
    type PolicyObs: Clone + Send + Batchable + 'static;
    /// The policy's action distribution type.
    type PolicyAD: Clone + Send + Batchable;
    /// The policy's action type.
    type PolicyAction: Clone + Send + Batchable;
    /// Additional data as context for an agent's action.
    type ActionContext: ItemLazy + Clone + Send + 'static;
    /// The state of the parameterized policy.
    type PolicyState: Clone + Send + PolicyState<Self::Backend> + 'static;

    /// The learning agent.
    type LearningAgent: PolicyLearner<
            Self::Backend,
            TrainContext = Self::TrainingOutput,
            InnerPolicy = Self::Policy,
        > + Send
        + 'static;
    /// The output data of a training step.
    type TrainingOutput: ItemLazy + Clone + Send;
}

/// Concrete type that implements the [ReinforcementLearningComponentsTypes](ReinforcementLearningComponentsTypes) trait.
pub struct RLComponentsMarker<B, E, EI, A> {
    _backend: PhantomData<B>,
    _env: PhantomData<E>,
    _env_init: PhantomData<EI>,
    _agent: PhantomData<A>,
}

impl<B, E, EI, A> RLComponentsTypes for RLComponentsMarker<B, E, EI, A>
where
    B: AutodiffBackend,
    E: Environment + 'static,
    EI: EnvironmentInit<E> + Send + 'static,
    A: PolicyLearner<B> + Send + 'static,
    A::TrainContext: ItemLazy + Clone + Send,
    A::InnerPolicy: Policy<B> + Send,
    <A::InnerPolicy as Policy<B>>::Observation: Batchable + Clone + Send,
    <A::InnerPolicy as Policy<B>>::ActionDistribution: Batchable + Clone + Send,
    <A::InnerPolicy as Policy<B>>::Action: Batchable + Clone + Send,
    <A::InnerPolicy as Policy<B>>::ActionContext: ItemLazy + Clone + Send + 'static,
    <A::InnerPolicy as Policy<B>>::PolicyState: Clone + Send,
    E::State: Into<<A::InnerPolicy as Policy<B>>::Observation> + Clone + Send + 'static,
    E::Action: From<<A::InnerPolicy as Policy<B>>::Action>
        + Into<<A::InnerPolicy as Policy<B>>::Action>
        + Clone
        + Send
        + 'static,
{
    type Backend = B;
    type Env = E;
    type EnvInit = EI;
    type LearningAgent = A;
    type Policy = A::InnerPolicy;
    type PolicyObs = <A::InnerPolicy as Policy<B>>::Observation;
    type PolicyAD = <A::InnerPolicy as Policy<B>>::ActionDistribution;
    type PolicyAction = <A::InnerPolicy as Policy<B>>::Action;
    type ActionContext = <A::InnerPolicy as Policy<B>>::ActionContext;
    type PolicyState = <A::InnerPolicy as Policy<B>>::PolicyState;
    type TrainingOutput = A::TrainContext;
    type State = E::State;
    type Action = E::Action;
}

pub(crate) type RlPolicy<RLC> = <<RLC as RLComponentsTypes>::LearningAgent as PolicyLearner<
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
pub type RLAgentRecord<RLC> = <<RLC as RLComponentsTypes>::LearningAgent as PolicyLearner<
    <RLC as RLComponentsTypes>::Backend,
>>::Record;

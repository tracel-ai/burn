use std::marker::PhantomData;

use burn_rl::{Batchable, Environment, EnvironmentInit, Policy, PolicyLearner, PolicyState};

use crate::{AgentEvaluationEvent, AsyncProcessorTraining, ItemLazy, RLEvent};

/// All components used by the reinforcement learning paradigm, grouped in one trait.
pub trait RLComponentsTypes {
    /// The learning environment.
    type Env: Environment<State = Self::State, Action = Self::Action> + 'static;
    /// Specifies how to initialize the environment.
    type EnvInit: EnvironmentInit<Self::Env> + Send + 'static;
    /// The type of the environment state.
    type State: Into<<Self::Policy as Policy>::Observation> + Clone + Send + 'static;
    /// The type of the environment action.
    type Action: From<<Self::Policy as Policy>::Action>
        + Into<<Self::Policy as Policy>::Action>
        + Clone
        + Send
        + 'static;

    /// The policy used to take actions in the environment.
    type Policy: Policy<
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
    type PolicyState: Clone + Send + PolicyState + 'static;

    /// The learning agent.
    type LearningAgent: PolicyLearner<TrainContext = Self::TrainingOutput, InnerPolicy = Self::Policy>
        + Send
        + 'static;
    /// The output data of a training step.
    type TrainingOutput: ItemLazy + Clone + Send;
}

/// Concrete type that implements the [RLComponentsTypes](RLComponentsTypes) trait.
pub struct RLComponentsMarker<E, EI, A> {
    _env: PhantomData<E>,
    _env_init: PhantomData<EI>,
    _agent: PhantomData<A>,
}

impl<E, EI, A> RLComponentsTypes for RLComponentsMarker<E, EI, A>
where
    E: Environment + 'static,
    EI: EnvironmentInit<E> + Send + 'static,
    A: PolicyLearner + Send + 'static,
    A::TrainContext: ItemLazy + Clone + Send,
    A::InnerPolicy: Policy + Send,
    <A::InnerPolicy as Policy>::Observation: Batchable + Clone + Send,
    <A::InnerPolicy as Policy>::ActionDistribution: Batchable + Clone + Send,
    <A::InnerPolicy as Policy>::Action: Batchable + Clone + Send,
    <A::InnerPolicy as Policy>::ActionContext: ItemLazy + Clone + Send + 'static,
    <A::InnerPolicy as Policy>::PolicyState: Clone + Send,
    E::State: Into<<A::InnerPolicy as Policy>::Observation> + Clone + Send + 'static,
    E::Action: From<<A::InnerPolicy as Policy>::Action>
        + Into<<A::InnerPolicy as Policy>::Action>
        + Clone
        + Send
        + 'static,
{
    type Env = E;
    type EnvInit = EI;
    type LearningAgent = A;
    type Policy = A::InnerPolicy;
    type PolicyObs = <A::InnerPolicy as Policy>::Observation;
    type PolicyAD = <A::InnerPolicy as Policy>::ActionDistribution;
    type PolicyAction = <A::InnerPolicy as Policy>::Action;
    type ActionContext = <A::InnerPolicy as Policy>::ActionContext;
    type PolicyState = <A::InnerPolicy as Policy>::PolicyState;
    type TrainingOutput = A::TrainContext;
    type State = E::State;
    type Action = E::Action;
}

pub(crate) type RlPolicy<RLC> =
    <<RLC as RLComponentsTypes>::LearningAgent as PolicyLearner>::InnerPolicy;
/// The event processor type for reinforcement learning.
pub type RLEventProcessorType<RLC> = AsyncProcessorTraining<
    RLEvent<<RLC as RLComponentsTypes>::TrainingOutput, <RLC as RLComponentsTypes>::ActionContext>,
    AgentEvaluationEvent<<RLC as RLComponentsTypes>::ActionContext>,
>;
/// The record of the policy.
pub type RLPolicyRecord<RLC> =
    <<<RLC as RLComponentsTypes>::Policy as Policy>::PolicyState as PolicyState>::Record;
/// The record of the learning agent.
pub type RLAgentRecord<RLC> = <<RLC as RLComponentsTypes>::LearningAgent as PolicyLearner>::Record;

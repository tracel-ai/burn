use std::marker::PhantomData;

use burn_core::tensor::backend::AutodiffBackend;
use burn_rl::{Environment, LearnerAgent, Policy};

use crate::{AgentEvaluationEvent, AsyncProcessorTraining, ItemLazy, RLEvent};

/// All components used by the reinforcement learning paradigm, grouped in one trait.
pub trait ReinforcementLearningComponentsTypes {
    /// The backend used for training.
    type Backend: AutodiffBackend;
    /// The learning environement.
    type Env: Environment<State = Self::State, Action = Self::Action> + 'static;
    /// The learning agent.
    /// // TODO: type shit is weird here.
    type LearningAgent: LearnerAgent<
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
    type State: Into<<Self::Policy as Policy<Self::Backend>>::Input> + Clone;
    type Action: From<<Self::Policy as Policy<Self::Backend>>::Action>;
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
    A: LearnerAgent<B, TrainingOutput = TO> + Send + 'static,
    A::InnerPolicy: Policy<B, ActionContext = AC> + Send,
    TO: ItemLazy + Clone + Send,
    AC: ItemLazy + Clone + Send + 'static,
    E::State: Into<<A::InnerPolicy as Policy<B>>::Input> + Clone,
    E::Action: From<<A::InnerPolicy as Policy<B>>::Action>,
{
    type Backend = B;
    type Env = E;
    type LearningAgent = A;
    type Policy = A::InnerPolicy;
    type ActionContext = AC;
    type TrainingOutput = TO;
    type State = E::State;
    type Action = E::Action;
}

pub(crate) type RlPolicy<OC> =
    <<OC as ReinforcementLearningComponentsTypes>::LearningAgent as LearnerAgent<
        <OC as ReinforcementLearningComponentsTypes>::Backend,
    >>::InnerPolicy;
// pub(crate) type RlPolicyState<OC> =
//     <<<OC as ReinforcementLearningComponentsTypes>::LearningAgent as LearnerAgent<
//         <OC as ReinforcementLearningComponentsTypes>::Backend,
//         RlState<OC>,
//         RlAction<OC>,
//     >>::InnerPolicy as Policy<
//         <OC as ReinforcementLearningComponentsTypes>::Backend,
//         RlState<OC>,
//         RlAction<OC>,
//     >>::PolicyState;
pub(crate) type RlState<OC> =
    <<OC as ReinforcementLearningComponentsTypes>::Env as Environment>::State;
pub(crate) type RlAction<OC> =
    <<OC as ReinforcementLearningComponentsTypes>::Env as Environment>::Action;
/// The event processor type for reinforcement learning.
pub type RLEventProcessorType<OC> = AsyncProcessorTraining<
    RLEvent<
        <OC as ReinforcementLearningComponentsTypes>::TrainingOutput,
        <OC as ReinforcementLearningComponentsTypes>::ActionContext,
    >,
    AgentEvaluationEvent<<OC as ReinforcementLearningComponentsTypes>::ActionContext>,
>;

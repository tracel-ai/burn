use std::marker::PhantomData;

use burn_core::tensor::backend::AutodiffBackend;
use burn_rl::{Agent, Environment, LearnerAgent};

use crate::{
    ItemLazy,
    metric::rl_processor::{FullEventProcessorTrainingRl, RlAsyncProcessorTraining},
};

/// All components used by the supervised learning paradigm, grouped in one trait.
pub trait OffPolicyLearningComponentsTypes {
    /// The [LearningComponents](crate::LearningComponentsTypes) types for supervised learning.
    // type LC: LearningComponentsTypes;
    type Backend: AutodiffBackend;
    type Env: Environment + 'static;
    type LearningAgent: LearnerAgent<
            // TrainingBackend<Self::LC>,
            Self::Backend,
            Self::Env,
            TrainingOutput = Self::TrainingOutput,
            DecisionContext = Self::ActionContext,
        > + Send
        + 'static;
    type ActionContext: ItemLazy + Clone + Send + 'static;
    type TrainingOutput: ItemLazy + Clone + Send;
}

/// Concrete type that implements the [OffPolicyLearningComponentsTypes](OffPolicyLearningComponentsTypes) trait.
pub struct OffPolicyLearningComponentsMarker<B, E, A, TO, AC> {
    _learner_components: PhantomData<B>,
    _env: PhantomData<E>,
    _agent: PhantomData<A>,
    _training_output: PhantomData<TO>,
    _action_context: PhantomData<AC>,
}

impl<B, E, A, TO, AC> OffPolicyLearningComponentsTypes
    for OffPolicyLearningComponentsMarker<B, E, A, TO, AC>
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

// pub(crate) type RlPolicy<OC> = <<OC as OffPolicyLearningComponentsTypes>::LearningAgent as Agent<
//     TrainingBackend<<OC as OffPolicyLearningComponentsTypes>::LC>,
//     <OC as OffPolicyLearningComponentsTypes>::Env,
// >>::Policy;
pub(crate) type RlPolicy<OC> = <<OC as OffPolicyLearningComponentsTypes>::LearningAgent as Agent<
    <OC as OffPolicyLearningComponentsTypes>::Backend,
    <OC as OffPolicyLearningComponentsTypes>::Env,
>>::Policy;
pub(crate) type RlState<OC> = <<OC as OffPolicyLearningComponentsTypes>::Env as Environment>::State;
pub(crate) type RlAction<OC> =
    <<OC as OffPolicyLearningComponentsTypes>::Env as Environment>::Action;
/// The event processor type for supervised learning.
// pub type RlEventProcessor<OC> = AsyncProcessorTraining<
//     FullEventProcessorTraining<
//         RlItemTypesTrain<
//             <OC as OffPolicyLearningComponentsTypes>::ActionContext,
//             EpisodeSummary,
//             <OC as OffPolicyLearningComponentsTypes>::TrainingOutput,
//         >,
//         RlItemTypesInference<<OC as OffPolicyLearningComponentsTypes>::ActionContext>,
//     >,
// >;
pub type RlEventProcessor<OC> = RlAsyncProcessorTraining<
    FullEventProcessorTrainingRl<
        <OC as OffPolicyLearningComponentsTypes>::TrainingOutput,
        <OC as OffPolicyLearningComponentsTypes>::ActionContext,
    >,
>;

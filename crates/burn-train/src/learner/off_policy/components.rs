use std::{
    io::{Error, ErrorKind},
    marker::PhantomData,
};

use burn_core::{prelude::Backend, tensor::backend::AutodiffBackend};
use burn_rl::{Agent, Environment, LearnerAgent};

use crate::{
    AsyncProcessorTraining, FullEventProcessorTraining, ItemLazy, LearningComponentsTypes,
    TrainingBackend,
    metric::{EpisodeLengthInput, EpisodeLengthMetric, ExplorationRateInput, LossInput, LossMetric, MetricAdaptor},
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

// /// Concrete type that implements the [OffPolicyLearningComponentsTypes](OffPolicyLearningComponentsTypes) trait.
// pub struct OffPolicyLearningComponentsMarker<LC, E, A, TO, AC> {
//     _learner_components: PhantomData<LC>,
//     _env: PhantomData<E>,
//     _agent: PhantomData<A>,
//     _training_output: PhantomData<TO>,
//     _action_context: PhantomData<AC>,
// }

// impl<LC, E, A, TO, AC> OffPolicyLearningComponentsTypes
//     for OffPolicyLearningComponentsMarker<LC, E, A, TO, AC>
// where
//     LC: LearningComponentsTypes,
//     E: Environment + 'static,
//     A: LearnerAgent<TrainingBackend<LC>, E, TrainingOutput = TO, DecisionContext = AC>
//         + Send
//         + 'static,
//     TO: ItemLazy + Clone + Send,
//     AC: ItemLazy + Clone + Send,
// {
//     type LC = LC;
//     type Env = E;
//     type LearningAgent = A;
//     type ActionContext = AC;
//     type TrainingOutput = TO;
// }

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
pub type RlEventProcessor<OC> = AsyncProcessorTraining<
    FullEventProcessorTraining<
        RlItemTypesTrain<
            <OC as OffPolicyLearningComponentsTypes>::ActionContext,
            EpisodeSummary,
            <OC as OffPolicyLearningComponentsTypes>::TrainingOutput,
        >,
        RlItemTypesInference<<OC as OffPolicyLearningComponentsTypes>::ActionContext>,
    >,
>;

// pub enum RlItemTypesTrain<SO, ES, TO> {
//     Step(SO),
//     EpisodeSummary(ES),
//     TrainStep(TO),
// }

// impl<SO: ItemLazy, ES: ItemLazy, TO: ItemLazy> ItemLazy for RlItemTypesTrain<SO, ES, TO> {
//     type ItemSync = RlItemTypesTrain<SO::ItemSync, ES::ItemSync, TO::ItemSync>;

//     fn sync(self) -> Self::ItemSync {
//         match self {
//             RlItemTypesTrain::Step(action_context) => RlItemTypesTrain::Step(action_context.sync()),
//             RlItemTypesTrain::EpisodeSummary(summary) => {
//                 RlItemTypesTrain::EpisodeSummary(summary.sync())
//             }
//             RlItemTypesTrain::TrainStep(output) => RlItemTypesTrain::TrainStep(output.sync()),
//         }
//     }
// }

// macro_rules! impl_adaptor_proxy1 {
//     ($metric:ty) => {
//         impl<SO, ES, TO> Adaptor<$metric> for RlItemTypesTrain<SO, ES, TO>
//         where
//             SO: Adaptor<$metric>,
//         {
//             fn adapt(&self) -> Result<$metric, Error> {
//                 match self {
//                     RlItemTypesTrain::Step(inner) => inner.adapt(),
//                     RlItemTypesTrain::EpisodeSummary(_inner) => {
//                         Err(Error::new(ErrorKind::Unsupported, "step"))
//                     }
//                     RlItemTypesTrain::TrainStep(_inner) => {
//                         Err(Error::new(ErrorKind::Unsupported, "step"))
//                     }
//                 }
//             }
//         }
//     };
// }
// macro_rules! impl_adaptor_proxy2 {
//     ($metric:ty) => {
//         impl<SO, ES, TO> Adaptor<$metric> for RlItemTypesTrain<SO, ES, TO>
//         where
//             ES: Adaptor<$metric>,
//         {
//             fn adapt(&self) -> Result<$metric, Error> {
//                 match self {
//                     RlItemTypesTrain::Step(_inner) => {
//                         Err(Error::new(ErrorKind::Unsupported, "summary"))
//                     }
//                     RlItemTypesTrain::EpisodeSummary(inner) => inner.adapt(),
//                     RlItemTypesTrain::TrainStep(_inner) => {
//                         Err(Error::new(ErrorKind::Unsupported, "summary"))
//                     }
//                 }
//             }
//         }
//     };
// }
// macro_rules! impl_adaptor_proxy3 {
//     ($metric:ty) => {
//         impl<SO, ES, TO> Adaptor<$metric> for RlItemTypesTrain<SO, ES, TO>
//         where
//             TO: Adaptor<$metric>,
//         {
//             fn adapt(&self) -> Result<$metric, Error> {
//                 match self {
//                     RlItemTypesTrain::Step(inner) => {
//                         Err(Error::new(ErrorKind::Unsupported, "train"))
//                     }
//                     RlItemTypesTrain::EpisodeSummary(inner) => {
//                         Err(Error::new(ErrorKind::Unsupported, "train"))
//                     }
//                     RlItemTypesTrain::TrainStep(inner) => inner.adapt(),
//                 }
//             }
//         }
//     };
// }
// macro_rules! impl_adaptor_proxy3_backend {
//     ($metric:ident) => {
//         impl<B: Backend, SO, ES, TO> Adaptor<$metric<B>> for RlItemTypesTrain<SO, ES, TO>
//         where
//             TO: Adaptor<$metric<B>>,
//         {
//             fn adapt(&self) -> Result<$metric<B>, Error> {
//                 match self {
//                     RlItemTypesTrain::Step(_inner) => {
//                         Err(Error::new(ErrorKind::Unsupported, "train"))
//                     }
//                     RlItemTypesTrain::EpisodeSummary(_inner) => {
//                         Err(Error::new(ErrorKind::Unsupported, "train"))
//                     }
//                     RlItemTypesTrain::TrainStep(inner) => inner.adapt(),
//                 }
//             }
//         }
//     };
// }
// // TODO: that's bad...
// impl_adaptor_proxy2!(EpisodeLengthInput);
// impl_adaptor_proxy1!(ExplorationRateInput);
// impl_adaptor_proxy3_backend!(LossInput);

// impl<SO, TO> Adaptor<EpisodeLengthInput> for RlItemTypesTrain<SO, TO> {
//     fn adapt(&self) -> Result<EpisodeLengthInput, Error> {
//         match self {
//             RlItemTypesTrain::Step(_) => Err(Error::new(
//                 ErrorKind::Unsupported,
//                 "Episode length cannot be inferred from a single environment step",
//             )),
//             RlItemTypesTrain::EpisodeSummary(summary) => Ok(EpisodeLengthInput {
//                 ep_len: summary.episode_length as f64,
//             }),
//             RlItemTypesTrain::TrainStep(_) => Err(Error::new(
//                 ErrorKind::Unsupported,
//                 "Episode length cannot be inferred from a training step",
//             )),
//         }
//     }
// }

// impl<SO: Adaptor<ExplorationRateInput>, TO> Adaptor<ExplorationRateInput>
//     for RlItemTypesTrain<SO, TO>
// {
//     fn adapt(&self) -> Result<ExplorationRateInput, Error> {
//         match self {
//             RlItemTypesTrain::Step(step) => step.adapt(),
//             RlItemTypesTrain::EpisodeSummary(_) => Err(Error::new(
//                 ErrorKind::Unsupported,
//                 "Exploration rate cannot be inferred from a single environment step",
//             )),
//             RlItemTypesTrain::TrainStep(_) => Err(Error::new(
//                 ErrorKind::Unsupported,
//                 "Exploration rate cannot be inferred from a training step",
//             )),
//         }
//     }
// }

// impl<B: Backend, SO, TO: Adaptor<LossInput<B>>> Adaptor<LossInput<B>> for RlItemTypesTrain<SO, TO> {
//     fn adapt(&self) -> Result<LossInput<B>, Error> {
//         match self {
//             RlItemTypesTrain::Step(_) => Err(Error::new(
//                 ErrorKind::Unsupported,
//                 "Loss cannot be inferred from a single environment step",
//             )),
//             RlItemTypesTrain::EpisodeSummary(_) => Err(Error::new(
//                 ErrorKind::Unsupported,
//                 "Loss cannot be inferred from an episode",
//             )),

//             RlItemTypesTrain::TrainStep(output) => output.adapt(),
//         }
//     }
// }

pub enum RlItemTypesInference<SO> {
    Step(SO),
    EpisodeSummary(EpisodeSummary),
}

impl<SO: ItemLazy> ItemLazy for RlItemTypesInference<SO> {
    type ItemSync = RlItemTypesInference<SO::ItemSync>;

    fn sync(self) -> Self::ItemSync {
        match self {
            RlItemTypesInference::Step(context) => {
                // Returns the synced context wrapped in the synced enum variant
                RlItemTypesInference::Step(context.sync())
            }
            RlItemTypesInference::EpisodeSummary(summary) => {
                // Returns the synced summary wrapped in the synced enum variant
                RlItemTypesInference::EpisodeSummary(summary.sync())
            }
        }
    }
}

// impl<SO> Adaptor<EpisodeLengthInput> for RlItemTypesInference<SO> {
//     fn adapt(&self) -> Result<EpisodeLengthInput, Error> {
//         match self {
//             RlItemTypesInference::Step(_) => Err(Error::new(
//                 ErrorKind::Unsupported,
//                 "Episode length cannot be inferred from single environment step",
//             )),
//             RlItemTypesInference::EpisodeSummary(summary) => Ok(EpisodeLengthInput {
//                 ep_len: summary.episode_length as f64,
//             }),
//         }
//     }
// }

// impl<SO: Adaptor<ExplorationRateInput>> Adaptor<ExplorationRateInput> for RlItemTypesInference<SO> {
//     fn adapt(&self) -> Result<ExplorationRateInput, Error> {
//         match self {
//             RlItemTypesInference::Step(step) => step.adapt(),
//             RlItemTypesInference::EpisodeSummary(_) => Err(Error::new(
//                 ErrorKind::Unsupported,
//                 "Exploration rate cannot be inferred from a single environment step",
//             )),
//         }
//     }
// }

pub struct EpisodeSummary {
    pub episode_length: usize,
    pub total_reward: f64,
}

impl ItemLazy for EpisodeSummary {
    type ItemSync = EpisodeSummary;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl MetricAdaptor<EpisodeLengthMetric> for EpisodeSummary {
    fn adapt(&self) -> EpisodeLengthInput {
        EpisodeLengthInput {
            ep_len: self.episode_length as f64,
        }
    }
}

use derive_new::new;

use burn_core::{prelude::*, record::Record, tensor::backend::AutodiffBackend};

use crate::TransitionBatch;

/// An action along with additional context about the decision.
#[derive(Clone, new)]
pub struct ActionContext<A, C> {
    /// The context.
    pub context: C,
    /// The action.
    pub action: A,
}

/// The state of a policy.
pub trait PolicyState<B: Backend> {
    /// The type of the record.
    type Record: Record<B>;

    /// Convert the state to a record.
    fn into_record(self) -> Self::Record;
    /// Load the state from a record.
    fn load_record(&self, record: Self::Record) -> Self;
}

/// Trait for a RL policy.
pub trait Policy<B: Backend>: Clone {
    /// The observation given as input to the policy.
    type Observation;
    /// The action distribution parameters defining how the action will be sampled.
    type ActionDistribution;
    /// The action.
    type Action;

    /// Additional context on the policy's decision.
    type ActionContext;
    /// The current parameterization of the policy.
    type PolicyState: PolicyState<B>;

    /// Produces the action distribution from a batch of observations.
    fn forward(&mut self, obs: Self::Observation) -> Self::ActionDistribution;
    /// Gives the action from a batch of observations.
    fn action(
        &mut self,
        obs: Self::Observation,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>);

    /// Update the policy's parameters.
    fn update(&mut self, update: Self::PolicyState);
    /// Returns the current parameterization.
    fn state(&self) -> Self::PolicyState;

    /// Loads the policy parameters from a record.
    fn load_record(self, record: <Self::PolicyState as PolicyState<B>>::Record) -> Self;
}

/// Trait for a type that can be batched and unbatched (split).
pub trait Batchable: Sized {
    /// Create a batch from a list of items.
    fn batch(value: Vec<Self>) -> Self;
    /// Create a list from batched items.
    fn unbatch(self) -> Vec<Self>;
}

/// A training output.
pub struct RLTrainOutput<TO, P> {
    /// The policy.
    pub policy: P,
    /// The item.
    pub item: TO,
}

/// Batched transitions for a PolicyLearner.
pub type LearnerTransitionBatch<B> = TransitionBatch<B>;

/// Learner for a policy.
pub trait PolicyLearner<B>
where
    B: AutodiffBackend,
    <Self::InnerPolicy as Policy<B>>::Observation: Clone + Batchable,
    <Self::InnerPolicy as Policy<B>>::ActionDistribution: Clone + Batchable,
    <Self::InnerPolicy as Policy<B>>::Action: Clone + Batchable,
{
    /// Additional context of a training step.
    type TrainContext;
    /// The policy to train.
    type InnerPolicy: Policy<B>;
    /// The record of the learner.
    type Record: Record<B>;

    /// Execute a training step on the policy.
    fn train(
        &mut self,
        input: LearnerTransitionBatch<B>,
    ) -> RLTrainOutput<Self::TrainContext, <Self::InnerPolicy as Policy<B>>::PolicyState>;
    /// Returns the learner's current policy.
    fn policy(&self) -> Self::InnerPolicy;
    /// Update the learner's policy.
    fn update_policy(&mut self, update: Self::InnerPolicy);

    /// Convert the learner's state into a record.
    fn record(&self) -> Self::Record;
    /// Load the learner's state from a record.
    fn load_record(self, record: Self::Record) -> Self;
}

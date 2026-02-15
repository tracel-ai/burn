#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! A library for training reinforcement learning agents.

/// Module for implementing an environment.
pub mod environment;
/// Module for implementing a policy.
pub mod policy;
/// Transition buffer.
pub mod transition_buffer;

pub use environment::*;
pub use policy::*;
pub use transition_buffer::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArray<f32>;

#[cfg(test)]
pub(crate) mod tests {
    use crate::{Batchable, Policy, PolicyState, TestBackend};

    use burn_core::record::Record;
    use burn_core::{self as burn};

    /// Mock policy for testing
    ///
    /// Calling `forward()` with a [MockObservation](MockObservation) (list of f32) returns a [MockActionDistribution](MockActionDistribution)
    /// containing a list of 0s of the same length as the observation.
    ///
    /// Calling `action()` with a [MockObservation](MockObservation) (list of f32) returns a [MockAction](MockAction) with a list of actions of the same length as the observation.
    /// The actions are all 1 if the call is requested as deterministic, or else 0.
    #[derive(Clone)]
    pub(crate) struct MockPolicy {}

    impl MockPolicy {
        pub fn new() -> Self {
            Self {}
        }
    }

    impl Policy<TestBackend> for MockPolicy {
        type Observation = MockObservation;
        type ActionDistribution = MockActionDistribution;
        type Action = MockAction;
        type ActionContext = MockActionContext;
        type PolicyState = MockPolicyState;

        fn forward(&mut self, obs: Self::Observation) -> Self::ActionDistribution {
            let mut dists = vec![];

            for _ in obs.0 {
                dists.push(MockActionDistribution(vec![0.]));
            }
            MockActionDistribution::batch(dists)
        }

        fn action(
            &mut self,
            obs: Self::Observation,
            deterministic: bool,
        ) -> (Self::Action, Vec<Self::ActionContext>) {
            let mut actions = vec![];
            let mut contexts = vec![];

            for _ in obs.0 {
                if deterministic {
                    actions.push(MockAction(vec![1]));
                } else {
                    actions.push(MockAction(vec![0]));
                }
                contexts.push(MockActionContext);
            }

            (MockAction::batch(actions), contexts)
        }

        fn update(&mut self, _update: Self::PolicyState) {}

        fn state(&self) -> Self::PolicyState {
            MockPolicyState
        }

        fn load_record(
            self,
            _record: <Self::PolicyState as PolicyState<TestBackend>>::Record,
        ) -> Self {
            self
        }
    }

    /// Mock observation for testing represented as a vector of f32. Can call `batch()` and `unbatch` on it.
    #[derive(Clone)]
    pub(crate) struct MockObservation(pub Vec<f32>);

    /// Mock action for testing represented as a vector of i32. Can call `batch()` and `unbatch` on it.
    #[derive(Clone)]
    pub(crate) struct MockAction(pub Vec<i32>);

    /// Mock action distribution for testing represented as a vector of i32. Can call `batch()` and `unbatch` on it.
    #[derive(Clone)]
    pub(crate) struct MockActionDistribution(Vec<f32>);

    #[derive(Clone)]
    pub(crate) struct MockActionContext;

    /// Mock policy state for testing represented as an arbitrary `usize` that has no effect on the policy.
    #[derive(Clone)]
    pub(crate) struct MockPolicyState;

    #[derive(Clone, Record)]
    pub(crate) struct MockRecord {
        item: usize,
    }

    impl PolicyState<TestBackend> for MockPolicyState {
        type Record = MockRecord;

        fn into_record(self) -> Self::Record {
            MockRecord { item: 0 }
        }

        fn load_record(&self, _record: Self::Record) -> Self {
            self.clone()
        }
    }

    impl Batchable for MockObservation {
        fn batch(items: Vec<Self>) -> Self {
            MockObservation(items.iter().flat_map(|m| m.0.clone()).collect())
        }

        fn unbatch(self) -> Vec<Self> {
            vec![MockObservation(self.0)]
        }
    }

    impl Batchable for MockAction {
        fn batch(items: Vec<Self>) -> Self {
            MockAction(items.iter().flat_map(|m| m.0.clone()).collect())
        }

        fn unbatch(self) -> Vec<Self> {
            let mut actions = vec![];
            for a in self.0 {
                actions.push(MockAction(vec![a]));
            }
            actions
        }
    }

    impl Batchable for MockActionDistribution {
        fn batch(items: Vec<Self>) -> Self {
            MockActionDistribution(items.iter().flat_map(|m| m.0.clone()).collect())
        }

        fn unbatch(self) -> Vec<Self> {
            let mut dists = vec![];
            for _ in self.0 {
                dists.push(MockActionDistribution(vec![0.]));
            }
            dists
        }
    }
}

mod async_runner;
mod base;

pub use async_runner::*;
pub use base::*;

#[cfg(test)]
pub(crate) mod tests {
    use burn_rl::{Batchable, Environment, EnvironmentInit, Policy, PolicyState};

    use crate::tests::TestAutodiffBackend;
    use crate::{
        AgentEvaluationEvent, EventProcessorTraining, ItemLazy, RLComponentsTypes, RLEvent,
    };
    use burn_rl::{LearnerTransitionBatch, PolicyLearner, RLTrainOutput, StepResult};

    /// Mock policy for testing
    ///
    /// Calling `forward()` with a [MockObservation](MockObservation) (list of f32) returns a [MockActionDistribution](MockActionDistribution)
    /// containing a list of 0s of the same length as the observation.
    ///
    /// Calling `action()` with a [MockObservation](MockObservation) (list of f32) returns a [MockPolicyAction](MockPolicyAction) with a list of actions of the same length as the observation.
    /// The actions are all 1 if the call is requested as deterministic, or else 0.

    #[derive(Clone)]
    pub(crate) struct MockPolicy(pub usize);

    impl Policy<TestAutodiffBackend> for MockPolicy {
        type Observation = MockObservation;
        type ActionDistribution = MockActionDistribution;
        type Action = MockPolicyAction;
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
                    actions.push(MockPolicyAction(vec![1]));
                } else {
                    actions.push(MockPolicyAction(vec![0]));
                }
                contexts.push(MockActionContext);
            }

            (MockPolicyAction::batch(actions), contexts)
        }

        fn update(&mut self, update: Self::PolicyState) {
            self.0 = update.0;
        }

        fn state(&self) -> Self::PolicyState {
            MockPolicyState(self.0)
        }

        fn load_record(
            self,
            _record: <Self::PolicyState as PolicyState<TestAutodiffBackend>>::Record,
        ) -> Self {
            self
        }
    }

    /// Mock observation for testing represented as a vector of f32. Can call `batch()` and `unbatch` on it.
    #[derive(Clone)]
    pub(crate) struct MockObservation(pub Vec<f32>);

    /// Mock action for testing represented as a vector of i32. Can call `batch()` and `unbatch` on it.
    #[derive(Clone)]
    pub(crate) struct MockPolicyAction(pub Vec<i32>);

    /// Mock action distribution for testing represented as a vector of i32. Can call `batch()` and `unbatch` on it.
    #[derive(Clone)]
    pub(crate) struct MockActionDistribution(Vec<f32>);

    #[derive(Clone)]
    pub(crate) struct MockActionContext;

    /// Mock policy state for testing represented as an arbitrary `usize` that has no effect on the policy.
    #[derive(Clone)]
    pub(crate) struct MockPolicyState(pub usize);

    impl PolicyState<TestAutodiffBackend> for MockPolicyState {
        type Record = ();

        fn into_record(self) -> Self::Record {
            ()
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

    impl Batchable for MockPolicyAction {
        fn batch(items: Vec<Self>) -> Self {
            MockPolicyAction(items.iter().flat_map(|m| m.0.clone()).collect())
        }

        fn unbatch(self) -> Vec<Self> {
            let mut actions = vec![];
            for a in self.0 {
                actions.push(MockPolicyAction(vec![a]));
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

    /// Mock environment for testing
    #[derive(Clone)]
    pub(crate) struct MockEnv {
        counter: usize,
    }

    #[derive(Clone, Debug)]
    pub(crate) struct MockState;

    #[derive(Clone, Debug)]
    pub(crate) struct MockAction(pub i32);

    impl From<MockState> for MockObservation {
        fn from(_value: MockState) -> Self {
            MockObservation(vec![0.])
        }
    }

    impl From<MockPolicyAction> for MockAction {
        fn from(value: MockPolicyAction) -> Self {
            MockAction(value.0[0])
        }
    }

    impl From<MockAction> for MockPolicyAction {
        fn from(value: MockAction) -> Self {
            MockPolicyAction(vec![value.0])
        }
    }

    impl ItemLazy for MockActionContext {
        type ItemSync = MockActionContext;

        fn sync(self) -> Self::ItemSync {
            self
        }
    }

    impl MockEnv {
        fn new() -> Self {
            Self { counter: 0 }
        }
    }

    impl Environment for MockEnv {
        type State = MockState;
        type Action = MockAction;
        const MAX_STEPS: usize = 5;

        fn reset(&mut self) {
            self.counter = 0;
        }

        fn step(&mut self, _action: Self::Action) -> StepResult<Self::State> {
            self.counter += 1;
            let done = self.counter >= Self::MAX_STEPS;

            burn_rl::StepResult {
                next_state: MockState,
                reward: 1.0,
                done,
                truncated: false,
            }
        }

        fn state(&self) -> Self::State {
            MockState
        }
    }

    /// Mock environment init for testing
    #[derive(Clone)]
    pub(crate) struct MockEnvInit;

    impl EnvironmentInit<MockEnv> for MockEnvInit {
        fn init(&self) -> MockEnv {
            MockEnv::new()
        }
    }

    // Mock RLComponentsTypes for testing
    pub(crate) struct MockRLComponents;

    impl RLComponentsTypes for MockRLComponents {
        type Backend = TestAutodiffBackend;
        type Env = MockEnv;
        type EnvInit = MockEnvInit;
        type State = MockState;
        type Action = MockAction;
        type Policy = MockPolicy;
        type PolicyObs = MockObservation;
        type PolicyAD = MockActionDistribution;
        type PolicyAction = MockPolicyAction;
        type ActionContext = MockActionContext;
        type PolicyState = MockPolicyState;
        type LearningAgent = MockLearningAgent;
        type TrainingOutput = ();
    }

    // Mock learning agent for testing
    #[derive(Clone)]
    pub(crate) struct MockLearningAgent;

    impl PolicyLearner<TestAutodiffBackend> for MockLearningAgent {
        type InnerPolicy = MockPolicy;
        type TrainContext = ();
        type Record = ();

        fn train(
            &mut self,
            _input: LearnerTransitionBatch<TestAutodiffBackend, Self::InnerPolicy>,
        ) -> RLTrainOutput<
            Self::TrainContext,
            <Self::InnerPolicy as Policy<TestAutodiffBackend>>::PolicyState,
        > {
            unimplemented!()
        }

        fn policy(&self) -> Self::InnerPolicy {
            unimplemented!()
        }

        fn update_policy(&mut self, _update: Self::InnerPolicy) {
            unimplemented!()
        }

        fn record(&self) -> Self::Record {
            unimplemented!()
        }

        fn load_record(self, _record: Self::Record) -> Self {
            unimplemented!()
        }
    }

    // Mock event processor for testing
    pub(crate) struct MockProcessor;

    impl
        EventProcessorTraining<
            RLEvent<(), MockActionContext>,
            AgentEvaluationEvent<MockActionContext>,
        > for MockProcessor
    {
        fn process_train(&mut self, _event: RLEvent<(), MockActionContext>) {
            // Mock process train
        }

        fn process_valid(&mut self, _event: AgentEvaluationEvent<MockActionContext>) {
            // Mock process valid
        }

        fn renderer(self) -> Box<dyn crate::renderer::MetricsRenderer> {
            unimplemented!()
        }
    }
}

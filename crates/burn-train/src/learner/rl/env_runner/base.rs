use std::marker::PhantomData;

use burn_core::data::dataloader::Progress;
use burn_core::{Tensor, prelude::Backend};
use burn_rl::Policy;
use burn_rl::Transition;
use burn_rl::{Environment, EnvironmentInit};

use crate::RLEvent;
use crate::{
    AgentEvaluationEvent, EpisodeSummary, EvaluationItem, EventProcessorTraining,
    RLEventProcessorType,
};
use crate::{Interrupter, RLComponentsTypes};

/// A trajectory, i.e. a list of ordered [TimeStep](TimeStep).
#[derive(Clone, new)]
pub struct Trajectory<B: Backend, S, A, C> {
    /// A list of ordered [TimeStep](TimeStep)s.
    pub timesteps: Vec<TimeStep<B, S, A, C>>,
}

/// A timestep debscribing an iteration of the state/decision process.
#[derive(Clone)]
pub struct TimeStep<B: Backend, S, A, C> {
    /// The environment id.
    pub env_id: usize,
    /// The [burn_rl::Transition](burn_rl::Transition).
    pub transition: Transition<B, S, A>,
    /// True if the environment reaches a terminal state.
    pub done: bool,
    /// The running length of the current episode.
    pub ep_len: usize,
    /// The running cumulative reward.
    pub cum_reward: f64,
    /// The action's context for this timestep.
    pub action_context: C,
}

pub(crate) type RLTimeStep<B, RLC> = TimeStep<
    B,
    <RLC as RLComponentsTypes>::State,
    <RLC as RLComponentsTypes>::Action,
    <RLC as RLComponentsTypes>::ActionContext,
>;

pub(crate) type RLTrajectory<B, RLC> = Trajectory<
    B,
    <RLC as RLComponentsTypes>::State,
    <RLC as RLComponentsTypes>::Action,
    <RLC as RLComponentsTypes>::ActionContext,
>;

/// Trait for a structure that implements an agent/environement interface.
pub trait AgentEnvLoop<BT: Backend, RLC: RLComponentsTypes> {
    /// Run a certain number of timesteps.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - The number of time_steps to run.
    /// * `processor` - An [crate::EventProcessorTraining](crate::EventProcessorTraining).
    /// * `interrupter` - An [crate::Interrupter](crate::Interrupter).
    /// * `num_steps` - The number of time_steps to run.
    /// * `progress` - A mutable reference to the learning progress.
    ///
    /// # Returns
    ///
    /// A list of ordered timesteps.
    fn run_steps(
        &mut self,
        num_steps: usize,
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<RLTimeStep<BT, RLC>>;
    /// Run a certain number of episodes.
    ///
    /// # Arguments
    ///
    /// * `num_episodes` - The number of episodes to run.
    /// * `processor` - An [crate::EventProcessorTraining](crate::EventProcessorTraining).
    /// * `interrupter` - An [crate::Interrupter](crate::Interrupter).
    /// * `progress` - A mutable reference to the learning progress.
    ///
    /// # Returns
    ///
    /// A list of ordered timesteps.
    fn run_episodes(
        &mut self,
        num_episodes: usize,
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<RLTrajectory<BT, RLC>>;
    /// Update the runner's agent.
    fn update_policy(&mut self, update: RLC::PolicyState);
    /// Get the state of the runner's agent.
    fn policy(&self) -> RLC::PolicyState;
}

/// A simple, synchronized agent/environement interface.
pub struct AgentEnvBaseLoop<B: Backend, RLC: RLComponentsTypes> {
    env: RLC::Env,
    eval: bool,
    agent: RLC::Policy,
    deterministic: bool,
    current_reward: f64,
    run_num: usize,
    step_num: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend, RLC: RLComponentsTypes> AgentEnvBaseLoop<B, RLC> {
    /// Create a new base runner.
    pub fn new(
        env_init: RLC::EnvInit,
        agent: RLC::Policy,
        eval: bool,
        deterministic: bool,
    ) -> Self {
        let mut env = env_init.init();
        env.reset();

        Self {
            env,
            eval,
            agent: agent.clone(),
            deterministic,
            current_reward: 0.0,
            run_num: 0,
            step_num: 0,
            _backend: PhantomData,
        }
    }
}

impl<BT, RLC> AgentEnvLoop<BT, RLC> for AgentEnvBaseLoop<BT, RLC>
where
    BT: Backend,
    RLC: RLComponentsTypes,
{
    fn run_steps(
        &mut self,
        num_steps: usize,
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<RLTimeStep<BT, RLC>> {
        let mut items = vec![];
        let device = Default::default();
        for _ in 0..num_steps {
            let state = self.env.state();
            let (action, context) = self.agent.action(state.clone().into(), self.deterministic);

            let step_result = self.env.step(RLC::Action::from(action.clone()));

            self.current_reward += step_result.reward;
            self.step_num += 1;

            let transition = Transition::new(
                state.clone(),
                step_result.next_state,
                RLC::Action::from(action),
                Tensor::from_data([step_result.reward], &device),
                Tensor::from_data(
                    [(step_result.done || step_result.truncated) as i32 as f64],
                    &device,
                ),
            );
            items.push(TimeStep {
                env_id: 0,
                transition,
                done: step_result.done,
                ep_len: self.step_num,
                cum_reward: self.current_reward,
                action_context: context[0].clone(),
            });

            if !self.eval {
                progress.items_processed += 1;
                processor.process_train(RLEvent::TimeStep(EvaluationItem::new(
                    context[0].clone(),
                    progress.clone(),
                    None,
                )));

                if step_result.done {
                    processor.process_train(RLEvent::EpisodeEnd(EvaluationItem::new(
                        EpisodeSummary {
                            episode_length: self.step_num,
                            cum_reward: self.current_reward,
                        },
                        progress.clone(),
                        None,
                    )));
                }
            }

            if interrupter.should_stop() {
                break;
            }

            if step_result.done || step_result.truncated {
                self.env.reset();
                self.current_reward = 0.;
                self.step_num = 0;
                self.run_num += 1;
            }
        }
        items
    }

    fn update_policy(&mut self, update: RLC::PolicyState) {
        self.agent.update(update);
    }

    fn run_episodes(
        &mut self,
        num_episodes: usize,
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<RLTrajectory<BT, RLC>> {
        self.env.reset();

        let mut items = vec![];
        for ep in 0..num_episodes {
            let mut steps = vec![];
            loop {
                let step = self.run_steps(1, processor, interrupter, progress)[0].clone();
                steps.push(step.clone());

                if self.eval {
                    processor.process_valid(AgentEvaluationEvent::TimeStep(EvaluationItem::new(
                        step.action_context.clone(),
                        Progress::new(steps.len() + 1, steps.len() + 1),
                        None,
                    )));

                    if step.done {
                        processor.process_valid(AgentEvaluationEvent::EpisodeEnd(
                            EvaluationItem::new(
                                EpisodeSummary {
                                    episode_length: step.ep_len,
                                    cum_reward: step.cum_reward,
                                },
                                Progress::new(ep + 1, num_episodes),
                                None,
                            ),
                        ));
                    }
                }

                if interrupter.should_stop() || step.done {
                    break;
                }
            }
            items.push(Trajectory::new(steps));

            if interrupter.should_stop() {
                break;
            }
        }
        items
    }

    fn policy(&self) -> RLC::PolicyState {
        self.agent.state()
    }
}

#[cfg(test)]
mod tests {
    use crate::{AsyncProcessorTraining, TestBackend};

    use crate::learner::tests::{
        MockEnvInit, MockPolicy, MockPolicyState, MockProcessor, MockRLComponents,
    };

    use super::*;

    fn setup(
        state: usize,
        eval: bool,
        deterministic: bool,
    ) -> AgentEnvBaseLoop<TestBackend, MockRLComponents> {
        let env_init = MockEnvInit;
        let agent = MockPolicy(state);
        AgentEnvBaseLoop::<TestBackend, MockRLComponents>::new(env_init, agent, eval, deterministic)
    }

    #[test]
    fn test_policy_returns_agent_state() {
        let runner = setup(1000, false, false);
        let policy_state = runner.policy();
        assert_eq!(policy_state.0, 1000);
    }

    #[test]
    fn test_update_policy() {
        let mut runner = setup(0, false, false);

        runner.update_policy(MockPolicyState(1));
        assert_eq!(runner.policy().0, 1);
    }

    #[test]
    fn run_steps_returns_requested_number() {
        let mut runner = setup(0, false, false);
        let mut processor = AsyncProcessorTraining::new(MockProcessor);
        let mut interrupter = Interrupter::new();
        let mut progress = Progress {
            items_processed: 0,
            items_total: 1,
        };

        let steps = runner.run_steps(1, &mut processor, &mut interrupter, &mut progress);
        assert_eq!(steps.len(), 1);
        let steps = runner.run_steps(8, &mut processor, &mut interrupter, &mut progress);
        assert_eq!(steps.len(), 8);
    }

    #[test]
    fn run_episodes_returns_requested_number() {
        let mut runner = setup(0, false, false);
        let mut processor = AsyncProcessorTraining::new(MockProcessor);
        let mut interrupter = Interrupter::new();
        let mut progress = Progress {
            items_processed: 0,
            items_total: 1,
        };

        let trajectories = runner.run_episodes(1, &mut processor, &mut interrupter, &mut progress);
        assert_eq!(trajectories.len(), 1);
        assert_ne!(trajectories[0].timesteps.len(), 0);
        let trajectories = runner.run_episodes(8, &mut processor, &mut interrupter, &mut progress);
        assert_eq!(trajectories.len(), 8);
        for i in 0..8 {
            assert_ne!(trajectories[i].timesteps.len(), 0);
        }
    }
}

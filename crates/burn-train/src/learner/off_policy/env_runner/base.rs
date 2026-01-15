use std::marker::PhantomData;

use burn_core::{Tensor, prelude::Backend};
use burn_rl::EnvAction;
use burn_rl::EnvState;
use burn_rl::Environment;
use burn_rl::Policy;
use burn_rl::Transition;

use crate::RLEventProcessorType;
use crate::RlAction;
use crate::RlPolicy;
use crate::RlState;
use crate::{Interrupter, ReinforcementLearningComponentsTypes};

/// A trajectory, i.e. a list of ordered [TimeStep](TimeStep)s.
#[derive(Clone)]
pub struct Trajectory<B: Backend, C> {
    // TODO : Change TimeStep to Transition. run_episodes should always return trajectories, even when async.
    // Probably need a step_sender/receiver and trajectory_sender/receiver in that case.
    /// A list of ordered [TimeStep](TimeStep)s.
    pub timesteps: Vec<TimeStep<B, C>>,
}

/// A timestep debscribing an iteration of the state/decision process.
#[derive(Clone)]
pub struct TimeStep<B: Backend, C> {
    /// The environment id.
    pub env_id: usize,
    /// The [burn_rl::Transition](burn_rl::Transition).
    pub transition: Transition<B>,
    /// True if the environement reaches a terminal state.
    pub done: bool,
    /// The running length of the current episode.
    pub ep_len: usize,
    /// The running cumulative reward.
    pub cum_reward: f64,
    /// The action's context for this timestep.
    pub action_context: C,
}

// TODO : Can Agent not be generic over backend?
// TODO : Remove BA and BT eventually and have only one Backend for the env runner.
/// Trait for a structure that implements an agent/environement interface.
pub trait EnvRunner<BT: Backend, OC: ReinforcementLearningComponentsTypes> {
    /// Start the runner.
    fn start(&mut self);
    /// Run a certain number of timesteps.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - The number of time_steps to run.
    /// * `deterministic` - If true, use the agent's deterministic policy, else use its stochastic policy.
    /// * `processor` - An [crate::EventProcessorTraining](crate::EventProcessorTraining).
    /// * `interrupter` - An [crate::Interrupter](crate::Interrupter).
    /// * `num_steps` - The number of time_steps to run.
    ///
    /// # Returns
    ///
    /// A list of ordered timesteps.
    fn run_steps(
        &mut self,
        num_steps: usize,
        deterministic: bool,
        processor: &mut RLEventProcessorType<OC>,
        interrupter: &Interrupter,
    ) -> Vec<TimeStep<BT, OC::ActionContext>>;
    /// Run a certain number of episodes.
    ///
    /// # Arguments
    ///
    /// * `num_episodes` - The number of episodes to run.
    /// * `deterministic` - If true, use the agent's deterministic policy, else use its stochastic policy.
    /// * `processor` - An [crate::EventProcessorTraining](crate::EventProcessorTraining).
    /// * `interrupter` - An [crate::Interrupter](crate::Interrupter).
    /// * `TODO:` - The number of time_steps to run.
    ///
    /// # Returns
    ///
    /// A list of ordered timesteps.
    fn run_episodes(
        &mut self,
        num_episodes: usize,
        deterministic: bool,
        processor: &mut RLEventProcessorType<OC>,
        interrupter: &Interrupter,
        global_iteration: usize,
        total_global_iteration: usize,
    ) -> Vec<Trajectory<BT, OC::ActionContext>>;
    /// Update the runner's agent's policy
    fn update_policy(
        &mut self,
        update: <RlPolicy<OC> as Policy<OC::Backend, RlState<OC>, RlAction<OC>>>::PolicyState,
    );
}

/// A simple, synchronized agent/environement interface.
pub struct BaseRunner<B: Backend, E: Environment, A: Policy<B, E::State, E::Action>> {
    env: E,
    agent: A,
    current_reward: f64,
    run_num: usize,
    step_num: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend, E: Environment, A: Policy<B, E::State, E::Action>> BaseRunner<B, E, A> {
    /// Create a new base runner.
    pub fn new(agent: A) -> Self {
        Self {
            env: E::new(),
            agent: agent.clone(),
            current_reward: 0.0,
            run_num: 0,
            step_num: 0,
            _backend: PhantomData,
        }
    }
}

impl<BT: Backend, OC: ReinforcementLearningComponentsTypes> EnvRunner<BT, OC>
    for BaseRunner<OC::Backend, OC::Env, OC::Policy>
{
    fn start(&mut self) {
        self.env.reset();
    }

    fn run_steps(
        &mut self,
        num_steps: usize,
        deterministic: bool,
        _processor: &mut RLEventProcessorType<OC>, // TODO:
        _interrupter: &Interrupter,
    ) -> Vec<TimeStep<BT, OC::ActionContext>> {
        let mut items = vec![];
        let device = Default::default();
        for _ in 0..num_steps {
            let state = self.env.state();
            // Assume only 1 action is returned since only 1 state is sent
            // TODO : Should agent also have take_action() for single envs?
            let action_context = self.agent.action(&state.clone(), deterministic).clone();

            let step_result = self.env.step(action_context.action.clone());

            self.current_reward += step_result.reward;
            self.step_num += 1;

            let transition = Transition::new(
                state.to_tensor(&device),
                step_result.next_state.to_tensor(&device),
                action_context.action.to_tensor(&device),
                Tensor::from_data([step_result.reward as f64], &device),
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
                action_context: action_context.context,
            });

            if step_result.done || step_result.truncated {
                self.env.reset();
                println!(
                    "Run : {}      Ep. len. : {}      Reward : {}",
                    self.run_num, self.step_num, self.current_reward
                );
                self.current_reward = 0.;
                self.step_num = 0;
                self.run_num += 1;
            }
        }
        items
    }

    fn update_policy(
        &mut self,
        update: <RlPolicy<OC> as Policy<OC::Backend, RlState<OC>, RlAction<OC>>>::PolicyState,
    ) {
        self.agent.update(update);
    }

    fn run_episodes(
        &mut self,
        _num_episodes: usize,
        _deterministic: bool,
        _processor: &mut RLEventProcessorType<OC>,
        _interrupter: &Interrupter,
        _global_iteration: usize,
        _total_global_iteration: usize,
    ) -> Vec<Trajectory<BT, OC::ActionContext>> {
        // TODO:
        // let mut items = vec![];
        // for _ in 0..num_episodes {
        //     let mut steps = vec![];
        //     loop {
        //         let step = &self.run_steps(1, deterministic, processor, interrupter)[0];
        //         steps.push(step.clone());
        //         if step.done {
        //             break;
        //         }
        //     }
        //     items.push(EpisodeTrajectory { steps });
        // }
        // items
        todo!()
    }
}

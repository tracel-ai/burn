use std::marker::PhantomData;

use burn_core::{Tensor, prelude::Backend};
use burn_rl::EnvAction;
use burn_rl::EnvState;
use burn_rl::Transition;
use burn_rl::{Agent, Environment};

use crate::RlEventProcessor;
use crate::RlPolicy;
use crate::{Interrupter, OffPolicyLearningComponentsTypes, TrainingBackend};

#[derive(Clone)]
pub struct EpisodeTrajectory<B: Backend, C> {
    // TODO : Change EnvStep to Transition. run_episodes should always return trajectories, even when async.
    // Probably need a step_sender/receiver and trajectory_sender/receiver in that case.
    pub steps: Vec<EnvStep<B, C>>,
}

#[derive(Clone)]
pub struct EnvStep<B: Backend, C> {
    pub env_id: usize,
    pub transition: Transition<B>,
    pub done: bool,
    pub ep_len: usize,
    pub cum_reward: f64,
    pub action_context: C,
}

// TODO : Can Agent not be generic over backend?
// TODO : Remove BA and BT eventually and have only one Backend for the env runner.
pub trait EnvRunner<BT: Backend, OC: OffPolicyLearningComponentsTypes> {
    fn start(&mut self);
    fn run_steps(
        &mut self,
        num_steps: usize,
        deterministic: bool,
        processor: &mut RlEventProcessor<OC>,
        interrupter: &Interrupter,
    ) -> Vec<EnvStep<BT, OC::ActionContext>>;
    fn run_episodes(
        &mut self,
        num_episodes: usize,
        deterministic: bool,
        processor: &mut RlEventProcessor<OC>,
        interrupter: &Interrupter,
        global_iteration: usize,
        total_global_iteration: usize,
    ) -> Vec<EpisodeTrajectory<BT, OC::ActionContext>>;
    fn update_policy(&mut self, update: RlPolicy<OC>);
}

pub struct BaseRunner<B: Backend, E: Environment, A: Agent<B, E>> {
    env: E,
    agent: A,
    current_reward: f64,
    run_num: usize,
    step_num: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend, E: Environment, A: Agent<B, E>> BaseRunner<B, E, A> {
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

// impl<BT: Backend, OC: OffPolicyLearningComponentsTypes> EnvRunner<BT, OC>
//     for BaseRunner<TrainingBackend<OC::LC>, OC::Env, OC::LearningAgent>
impl<BT: Backend, OC: OffPolicyLearningComponentsTypes> EnvRunner<BT, OC>
    for BaseRunner<OC::Backend, OC::Env, OC::LearningAgent>
{
    fn start(&mut self) {
        self.env.reset();
    }

    fn run_steps(
        &mut self,
        num_steps: usize,
        deterministic: bool,
        processor: &mut RlEventProcessor<OC>,
        interrupter: &Interrupter,
    ) -> Vec<EnvStep<BT, OC::ActionContext>> {
        let mut items = vec![];
        let device = Default::default();
        for _ in 0..num_steps {
            let state = self.env.state();
            // Assume only 1 action is returned since only 1 state is sent
            // TODO : Should agent also have take_action() for single envs?
            let action_context = self.agent.take_action(state.clone(), deterministic).clone();

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
            items.push(EnvStep {
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

    fn update_policy(&mut self, update: RlPolicy<OC>) {
        self.agent.update_policy(update);
    }

    fn run_episodes(
        &mut self,
        num_episodes: usize,
        deterministic: bool,
        processor: &mut RlEventProcessor<OC>,
        interrupter: &Interrupter,
        global_iteration: usize,
        total_global_iteration: usize,
    ) -> Vec<EpisodeTrajectory<BT, OC::ActionContext>> {
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

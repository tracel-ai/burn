use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, Sender},
    thread::spawn,
};

use burn_core::{Tensor, data::dataloader::Progress, prelude::Backend, tensor::Device};
use burn_rl::EnvState;
use burn_rl::{Agent, AsyncAgent, Environment};
use burn_rl::{EnvAction, Transition};

use crate::{
    AgentEvaluationEvent, EnvRunner, EpisodeSummary, EventProcessorTraining, Interrupter,
    LearnerItem, RLEventProcessorType, ReinforcementLearningComponentsTypes, RlPolicy, RlState,
    TimeStep, Trajectory,
};

struct StepMessage<B: Backend, C> {
    step: TimeStep<B, C>,
    confirmation_sender: Sender<()>,
}

/// An asynchronous agent/environement interface.
pub struct AsyncEnvRunner<BT: Backend, OC: ReinforcementLearningComponentsTypes> {
    id: usize,
    agent: AsyncAgent<OC::Backend, OC::Env, OC::LearningAgent>,
    deterministic: bool,
    transition_device: Device<BT>,
    transition_receiver: Receiver<StepMessage<BT, OC::ActionContext>>,
    transition_sender: Sender<StepMessage<BT, OC::ActionContext>>,
}

impl<BT: Backend, OC: ReinforcementLearningComponentsTypes> AsyncEnvRunner<BT, OC> {
    /// Create a new asynchronous runner.
    pub fn new(
        id: usize,
        agent: AsyncAgent<OC::Backend, OC::Env, OC::LearningAgent>,
        deterministic: bool,
        transition_device: &Device<BT>,
    ) -> Self {
        let (transition_sender, transition_receiver) = std::sync::mpsc::channel();
        Self {
            id,
            agent: agent.clone(),
            deterministic,
            transition_device: transition_device.clone(),
            transition_receiver,
            transition_sender,
        }
    }
}

impl<BT: Backend, OC: ReinforcementLearningComponentsTypes> EnvRunner<BT, OC>
    for AsyncEnvRunner<BT, OC>
{
    fn start(&mut self) {
        let id = self.id;
        let mut agent = self.agent.clone();
        let deterministic = self.deterministic;
        let transition_sender = self.transition_sender.clone();
        let device = self.transition_device.clone();

        let mut current_reward = 0.0;
        let mut step_num = 0;

        // TODO : When running full episodes, dont block at every step, just block after episode ends (start blocking at 2nd episode end).
        spawn(move || {
            let mut env = OC::Env::new();
            env.reset();
            let (confirmation_sender, confirmation_receiver) = std::sync::mpsc::channel();
            confirmation_sender
                .send(())
                .expect("Can send initial confirmation message");
            loop {
                let state = env.state();
                let action_context = agent
                    .batch_take_action(Vec::<RlState<OC>>::from([state.clone()]), deterministic)[0]
                    .clone();

                let step_result = env.step(action_context.action.clone());

                current_reward += step_result.reward;
                step_num += 1;

                confirmation_receiver
                    .recv()
                    .expect("Can receive confirmation from main runner thread.");
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
                transition_sender
                    .send(StepMessage {
                        step: TimeStep {
                            env_id: id,
                            transition,
                            done: step_result.done,
                            ep_len: step_num,
                            cum_reward: current_reward,
                            action_context: action_context.context,
                        },
                        confirmation_sender: confirmation_sender.clone(),
                    })
                    .expect("Can send transition on channel");

                if step_result.done || step_result.truncated {
                    env.reset();
                    current_reward = 0.;
                    step_num = 0;
                }
            }
        });
    }

    fn run_steps(
        &mut self,
        num_steps: usize,
        _deterministic: bool,
        processor: &mut RLEventProcessorType<OC>,
        interrupter: &Interrupter,
    ) -> Vec<TimeStep<BT, OC::ActionContext>> {
        let mut items = vec![];
        for _ in 0..num_steps {
            let msg = self
                .transition_receiver
                .recv()
                .expect("Can receive transitions.");
            items.push(msg.step);
            msg.confirmation_sender
                .send(())
                .expect("Can send confirmation to env worker.")
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
        processor: &mut RLEventProcessorType<OC>,
        interrupter: &Interrupter,
        global_iteration: usize,
        total_global_iteration: usize,
    ) -> Vec<Trajectory<BT, OC::ActionContext>> {
        let mut items = vec![];
        for episode_num in 0..num_episodes {
            let mut steps = vec![];
            let mut step_num = 0;
            loop {
                let step = &self.run_steps(1, deterministic, processor, interrupter)[0];
                steps.push(step.clone());

                step_num += 1;
                processor.process_valid(AgentEvaluationEvent::TimeStep(LearnerItem::new(
                    step.action_context.clone(),
                    Progress::new(episode_num, num_episodes),
                    global_iteration,
                    total_global_iteration,
                    step_num,
                    None,
                )));

                if step.done {
                    processor.process_valid(AgentEvaluationEvent::EpisodeEnd(LearnerItem::new(
                        EpisodeSummary {
                            episode_length: step.ep_len,
                            cum_reward: step.cum_reward,
                        },
                        Progress::new(episode_num, num_episodes),
                        global_iteration,
                        total_global_iteration,
                        episode_num,
                        None,
                    )));
                    break;
                }
            }
            items.push(Trajectory { timesteps: steps });
        }
        items
    }
}

/// An asynchronous runner for multiple agent/environement interfaces.
pub struct AsyncEnvArrayRunner<BT: Backend, OC: ReinforcementLearningComponentsTypes> {
    num_envs: usize,
    agent: AsyncAgent<OC::Backend, OC::Env, OC::LearningAgent>,
    deterministic: bool,
    device: Device<BT>,
    transition_receiver: Receiver<StepMessage<BT, OC::ActionContext>>,
    transition_sender: Sender<StepMessage<BT, OC::ActionContext>>,
    current_trajectories: HashMap<usize, Vec<TimeStep<BT, OC::ActionContext>>>,
}

impl<BT: Backend, OC: ReinforcementLearningComponentsTypes> AsyncEnvArrayRunner<BT, OC> {
    /// Create a new asynchronous runner for multiple agent/environement interfaces.
    pub fn new(
        num_envs: usize,
        agent: AsyncAgent<OC::Backend, OC::Env, OC::LearningAgent>,
        deterministic: bool,
        device: &Device<BT>,
    ) -> Self {
        let (transition_sender, transition_receiver) = std::sync::mpsc::channel();
        Self {
            num_envs,
            agent: agent.clone(),
            deterministic,
            device: device.clone(),
            transition_receiver,
            transition_sender,
            current_trajectories: HashMap::default(),
        }
    }
}

impl<BT: Backend, OC: ReinforcementLearningComponentsTypes> EnvRunner<BT, OC>
    for AsyncEnvArrayRunner<BT, OC>
{
    // TODO: start() shouldn't exist.
    fn start(&mut self) {
        for i in 0..self.num_envs {
            let mut runner = AsyncEnvRunner::<BT, OC>::new(
                i,
                self.agent.clone(),
                self.deterministic,
                &self.device,
            );
            runner.transition_sender = self.transition_sender.clone();
            runner.start();
        }
    }

    fn run_steps(
        &mut self,
        num_steps: usize,
        _deterministic: bool,
        processor: &mut RLEventProcessorType<OC>,
        interrupter: &Interrupter,
    ) -> Vec<TimeStep<BT, OC::ActionContext>> {
        let mut items = vec![];
        for _ in 0..num_steps {
            let msg = self
                .transition_receiver
                .recv()
                .expect("Can receive transitions.");
            items.push(msg.step);
            msg.confirmation_sender
                .send(())
                .expect("Can send confirmation to env worker.");
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
        processor: &mut RLEventProcessorType<OC>,
        interrupter: &Interrupter,
        global_iteration: usize,
        total_global_iteration: usize,
    ) -> Vec<Trajectory<BT, OC::ActionContext>> {
        let mut items = vec![];
        loop {
            let step = &self.run_steps(1, deterministic, processor, interrupter)[0];
            self.current_trajectories
                .entry(step.env_id)
                .or_default()
                .push(step.clone());
            if step.done
                && let Some(steps) = self.current_trajectories.get_mut(&step.env_id)
            {
                items.push(Trajectory {
                    timesteps: steps.clone(),
                });
                steps.clear();
            }
            if items.len() >= num_episodes {
                break;
            }
        }
        items
    }
}

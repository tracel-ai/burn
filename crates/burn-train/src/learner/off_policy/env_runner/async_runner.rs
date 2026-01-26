use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, Sender},
    thread::spawn,
};

use burn_core::{Tensor, data::dataloader::Progress, prelude::Backend, tensor::Device};
use burn_rl::Transition;
use burn_rl::{AsyncPolicy, Environment};
use burn_rl::{AgentLearner, Policy};

use crate::{
    AgentEvaluationEvent, EnvRunner, EpisodeSummary, EvaluationItem, EventProcessorTraining,
    Interrupter, RLEvent, RLEventProcessorType, ReinforcementLearningComponentsTypes, RlPolicy,
    TimeStep, Trajectory,
};

struct StepMessage<B: Backend, S, A, C> {
    step: TimeStep<B, S, A, C>,
    confirmation_sender: Sender<()>,
}

pub(crate) type RLTimeStep<B, RLC> = TimeStep<
    B,
    <<RLC as ReinforcementLearningComponentsTypes>::Policy as Policy<
        <RLC as ReinforcementLearningComponentsTypes>::Backend,
    >>::Input,
    <<RLC as ReinforcementLearningComponentsTypes>::Policy as Policy<
        <RLC as ReinforcementLearningComponentsTypes>::Backend,
    >>::Action,
    <RLC as ReinforcementLearningComponentsTypes>::ActionContext,
>;

type RLStepMessage<B, RLC> = StepMessage<
    B,
    <<RLC as ReinforcementLearningComponentsTypes>::Policy as Policy<
        <RLC as ReinforcementLearningComponentsTypes>::Backend,
    >>::Input,
    <<RLC as ReinforcementLearningComponentsTypes>::Policy as Policy<
        <RLC as ReinforcementLearningComponentsTypes>::Backend,
    >>::Action,
    <RLC as ReinforcementLearningComponentsTypes>::ActionContext,
>;

/// An asynchronous agent/environement interface.
pub struct AsyncEnvRunner<BT: Backend, RLC: ReinforcementLearningComponentsTypes> {
    id: usize,
    eval: bool,
    agent: AsyncPolicy<RLC::Backend, RlPolicy<RLC>>,
    deterministic: bool,
    transition_device: Device<BT>,
    transition_receiver: Receiver<RLStepMessage<BT, RLC>>,
    transition_sender: Sender<RLStepMessage<BT, RLC>>,
}

impl<BT: Backend, RLC: ReinforcementLearningComponentsTypes> AsyncEnvRunner<BT, RLC> {
    /// Create a new asynchronous runner.
    pub fn new(
        id: usize,
        eval: bool,
        agent: AsyncPolicy<RLC::Backend, RlPolicy<RLC>>,
        deterministic: bool,
        transition_device: &Device<BT>,
    ) -> Self {
        let (transition_sender, transition_receiver) = std::sync::mpsc::channel();
        Self {
            id,
            eval,
            agent: agent.clone(),
            deterministic,
            transition_device: transition_device.clone(),
            transition_receiver,
            transition_sender,
        }
    }
}

impl<BT, RLC> EnvRunner<BT, RLC> for AsyncEnvRunner<BT, RLC>
where
    BT: Backend,
    RLC: ReinforcementLearningComponentsTypes,
    RLC::Policy: Send + 'static,
    <RLC::Policy as Policy<RLC::Backend>>::PolicyState: Send,
    <RLC::Policy as Policy<RLC::Backend>>::ActionContext: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Input: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Action: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Output: Send,
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
            let mut env = RLC::Env::new();
            env.reset();
            let (confirmation_sender, confirmation_receiver) = std::sync::mpsc::channel();
            confirmation_sender
                .send(())
                .expect("Can send initial confirmation message");
            loop {
                let state = env.state();
                let (action, context) = agent.action(state.clone().into(), deterministic);

                let step_result = env.step(action.clone().into());

                current_reward += step_result.reward;
                step_num += 1;

                confirmation_receiver
                    .recv()
                    .expect("Can receive confirmation from main runner thread.");
                let transition = Transition::new(
                    state.into(),
                    step_result.next_state.into(),
                    action,
                    Tensor::from_data([step_result.reward as f64], &device),
                    Tensor::from_data(
                        [(step_result.done || step_result.truncated) as i32 as f64],
                        &device,
                    ),
                );
                let res = transition_sender.send(StepMessage {
                    step: TimeStep {
                        env_id: id,
                        transition,
                        done: step_result.done,
                        ep_len: step_num,
                        cum_reward: current_reward,
                        action_context: context[0].clone(),
                    },
                    confirmation_sender: confirmation_sender.clone(),
                });

                match res {
                    Err(err) => {
                        log::error!("Error in env runner : {}", err);
                        break;
                    }
                    _ => (),
                }

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
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<RLTimeStep<BT, RLC>> {
        let mut items = vec![];
        for _ in 0..num_steps {
            let msg = self
                .transition_receiver
                .recv()
                .expect("Can receive transitions.");
            items.push(msg.step.clone());
            msg.confirmation_sender
                .send(())
                .expect("Can send confirmation to env worker.");

            if !self.eval {
                progress.items_processed += 1;
                processor.process_train(RLEvent::TimeStep(EvaluationItem::new(
                    msg.step.action_context,
                    progress.clone(),
                    None,
                )));

                if msg.step.done {
                    processor.process_train(RLEvent::EpisodeEnd(EvaluationItem::new(
                        EpisodeSummary {
                            episode_length: msg.step.ep_len,
                            cum_reward: msg.step.cum_reward,
                        },
                        progress.clone(),
                        None,
                    )));
                }
            }

            if interrupter.should_stop() {
                break;
            }
        }
        items
    }

    fn update_policy(&mut self, update: <RlPolicy<RLC> as Policy<RLC::Backend>>::PolicyState) {
        self.agent.update(update);
    }

    fn run_episodes(
        &mut self,
        num_episodes: usize,
        deterministic: bool,
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<
        Trajectory<
            BT,
            <RLC::Policy as Policy<RLC::Backend>>::Input,
            <RLC::Policy as Policy<RLC::Backend>>::Action,
            RLC::ActionContext,
        >,
    > {
        let mut items = vec![];
        for episode_num in 0..num_episodes {
            let mut steps = vec![];
            let mut step_num = 0;
            loop {
                let step =
                    self.run_steps(1, deterministic, processor, interrupter, progress)[0].clone();
                steps.push(step.clone());

                step_num += 1;

                if self.eval {
                    processor.process_valid(AgentEvaluationEvent::TimeStep(EvaluationItem::new(
                        step.action_context.clone(),
                        Progress::new(step_num, step_num),
                        None,
                    )));

                    if step.done {
                        processor.process_valid(AgentEvaluationEvent::EpisodeEnd(
                            EvaluationItem::new(
                                EpisodeSummary {
                                    episode_length: step.ep_len,
                                    cum_reward: step.cum_reward,
                                },
                                Progress::new(episode_num + 1, num_episodes),
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
        }
        items
    }
    
    fn policy(&self) -> <RlPolicy<RLC> as Policy<<RLC as ReinforcementLearningComponentsTypes>::Backend>>::PolicyState {
        self.agent.state()
    }
}

/// An asynchronous runner for multiple agent/environement interfaces.
pub struct AsyncEnvArrayRunner<BT: Backend, RLC: ReinforcementLearningComponentsTypes> {
    num_envs: usize,
    eval: bool,
    agent:
        AsyncPolicy<RLC::Backend, <RLC::LearningAgent as AgentLearner<RLC::Backend>>::InnerPolicy>,
    deterministic: bool,
    device: Device<BT>,
    transition_receiver: Receiver<RLStepMessage<BT, RLC>>,
    transition_sender: Sender<RLStepMessage<BT, RLC>>,
    current_trajectories: HashMap<usize, Vec<RLTimeStep<BT, RLC>>>,
}

impl<BT: Backend, RLC: ReinforcementLearningComponentsTypes> AsyncEnvArrayRunner<BT, RLC> {
    /// Create a new asynchronous runner for multiple agent/environement interfaces.
    pub fn new(
        num_envs: usize,
        eval: bool,
        agent: AsyncPolicy<RLC::Backend, RLC::Policy>,
        deterministic: bool,
        device: &Device<BT>,
    ) -> Self {
        let (transition_sender, transition_receiver) = std::sync::mpsc::channel();
        Self {
            num_envs,
            eval,
            agent: agent.clone(),
            deterministic,
            device: device.clone(),
            transition_receiver,
            transition_sender,
            current_trajectories: HashMap::default(),
        }
    }
}

impl<BT, RLC> EnvRunner<BT, RLC> for AsyncEnvArrayRunner<BT, RLC>
where
    BT: Backend,
    RLC: ReinforcementLearningComponentsTypes,
    RLC::Policy: Send + 'static,
    <RLC::Policy as Policy<RLC::Backend>>::PolicyState: Send,
    <RLC::Policy as Policy<RLC::Backend>>::ActionContext: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Input: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Action: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Output: Send,
{
    // TODO: start() shouldn't exist.
    fn start(&mut self) {
        for i in 0..self.num_envs {
            let mut runner = AsyncEnvRunner::<BT, RLC>::new(
                i,
                self.eval,
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
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<RLTimeStep<BT, RLC>> {
        let mut items = vec![];
        for _ in 0..num_steps {
            let msg = self
                .transition_receiver
                .recv()
                .expect("Can receive transitions.");
            items.push(msg.step.clone());
            msg.confirmation_sender
                .send(())
                .expect("Can send confirmation to env worker.");

            progress.items_processed += 1;

            if !self.eval {
                processor.process_train(RLEvent::TimeStep(EvaluationItem::new(
                    msg.step.action_context,
                    progress.clone(),
                    None,
                )));

                if msg.step.done {
                    processor.process_train(RLEvent::EpisodeEnd(EvaluationItem::new(
                        EpisodeSummary {
                            episode_length: msg.step.ep_len,
                            cum_reward: msg.step.cum_reward,
                        },
                        progress.clone(),
                        None,
                    )));
                }
            }

            if interrupter.should_stop() {
                break;
            }
        }
        items
    }

    fn update_policy(&mut self, update: <RlPolicy<RLC> as Policy<RLC::Backend>>::PolicyState) {
        self.agent.update(update);
    }

    fn run_episodes(
        &mut self,
        num_episodes: usize,
        deterministic: bool,
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<
        Trajectory<
            BT,
            <RLC::Policy as Policy<RLC::Backend>>::Input,
            <RLC::Policy as Policy<RLC::Backend>>::Action,
            RLC::ActionContext,
        >,
    > {
        let mut items = vec![];
        loop {
            let step = &self.run_steps(1, deterministic, processor, interrupter, progress)[0];
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
    
    fn policy(&self) -> <RlPolicy<RLC> as Policy<<RLC as ReinforcementLearningComponentsTypes>::Backend>>::PolicyState {
        self.agent.state()
    }
}

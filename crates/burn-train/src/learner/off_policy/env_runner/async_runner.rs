use rand::prelude::SliceRandom;
use std::{
    sync::mpsc::{Receiver, Sender},
    thread::spawn,
};

use burn_core::{Tensor, data::dataloader::Progress, prelude::Backend, tensor::Device};
use burn_rl::EnvironmentInit;
use burn_rl::Policy;
use burn_rl::Transition;
use burn_rl::{AsyncPolicy, Environment};

use crate::{
    AgentEnvLoop, AgentEvaluationEvent, EpisodeSummary, EvaluationItem, EventProcessorTraining,
    Interrupter, RLComponentsTypes, RLEvent, RLEventProcessorType, RLTimeStep, RLTrajectory,
    RlPolicy, TimeStep, Trajectory,
};

enum RequestMessage {
    Step(),
    Episode(),
}

/// An asynchronous agent/environement interface.
pub struct AgentEnvAsyncLoop<BT: Backend, RLC: RLComponentsTypes> {
    env_init: RLC::EnvInit,
    id: usize,
    eval: bool,
    agent: AsyncPolicy<RLC::Backend, RlPolicy<RLC>>,
    deterministic: bool,
    transition_device: Device<BT>,
    transition_receiver: Receiver<RLTimeStep<BT, RLC>>,
    transition_sender: Sender<RLTimeStep<BT, RLC>>,
    trajectory_receiver: Receiver<RLTrajectory<BT, RLC>>,
    trajectory_sender: Sender<RLTrajectory<BT, RLC>>,
    request_sender: Option<Sender<RequestMessage>>,
}

impl<BT: Backend, RLC: RLComponentsTypes> AgentEnvAsyncLoop<BT, RLC> {
    /// Create a new asynchronous runner.
    pub fn new(
        env_init: RLC::EnvInit,
        id: usize,
        eval: bool,
        agent: AsyncPolicy<RLC::Backend, RlPolicy<RLC>>,
        deterministic: bool,
        transition_device: &Device<BT>,
    ) -> Self {
        let (transition_sender, transition_receiver) = std::sync::mpsc::channel();
        let (trajectory_sender, trajectory_receiver) = std::sync::mpsc::channel();
        Self {
            env_init,
            id,
            eval,
            agent: agent.clone(),
            deterministic,
            transition_device: transition_device.clone(),
            transition_receiver,
            transition_sender,
            trajectory_receiver,
            trajectory_sender,
            request_sender: None,
        }
    }
}

impl<BT, RLC> AgentEnvLoop<BT, RLC> for AgentEnvAsyncLoop<BT, RLC>
where
    BT: Backend,
    RLC: RLComponentsTypes,
    RLC::Policy: Send + 'static,
    <RLC::Policy as Policy<RLC::Backend>>::PolicyState: Send,
    <RLC::Policy as Policy<RLC::Backend>>::ActionContext: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Observation: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Action: Send,
    <RLC::Policy as Policy<RLC::Backend>>::ActionDistribution: Send,
{
    fn start(&mut self) {
        let id = self.id;
        let mut agent = self.agent.clone();
        let deterministic = self.deterministic;
        let transition_sender = self.transition_sender.clone();
        let trajectory_sender = self.trajectory_sender.clone();
        let device = self.transition_device.clone();
        let env_init = self.env_init.clone();

        let (request_sender, request_receiver) = std::sync::mpsc::channel();
        self.request_sender = Some(request_sender);

        let mut current_steps = vec![];
        let mut current_reward = 0.0;
        let mut step_num = 0;

        spawn(move || {
            let mut env = env_init.init();
            env.reset();

            let mut request_episode = false;
            loop {
                let state = env.state();
                let (action, context) = agent.action(state.clone().into(), deterministic);

                let env_action = RLC::Action::from(action);
                let step_result = env.step(env_action.clone());

                current_reward += step_result.reward;
                step_num += 1;

                let transition = Transition::new(
                    state.clone(),
                    step_result.next_state,
                    env_action,
                    Tensor::from_data([step_result.reward as f64], &device),
                    Tensor::from_data(
                        [(step_result.done || step_result.truncated) as i32 as f64],
                        &device,
                    ),
                );

                if !request_episode {
                    agent.decrement_agents(1);
                    let request = match request_receiver.recv() {
                        Ok(req) => req,
                        Err(err) => {
                            log::error!("Error in env runner : {}", err);
                            break;
                        }
                    };
                    agent.increment_agents(1);

                    match request {
                        RequestMessage::Step() => (),
                        RequestMessage::Episode() => request_episode = true,
                    }
                }

                let time_step = TimeStep {
                    env_id: id,
                    transition,
                    done: step_result.done,
                    ep_len: step_num,
                    cum_reward: current_reward,
                    action_context: context[0].clone(),
                };
                current_steps.push(time_step.clone());

                if !request_episode {
                    if let Err(err) = transition_sender.send(time_step) {
                        log::error!("Error in env runner : {}", err);
                        break;
                    }
                }

                if step_result.done || step_result.truncated {
                    if request_episode {
                        request_episode = false;
                        trajectory_sender
                            .send(Trajectory {
                                timesteps: current_steps.clone(),
                            })
                            .expect("Can send trajectory to main thread.");
                    }
                    current_steps.clear();

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
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<RLTimeStep<BT, RLC>> {
        let mut items = vec![];
        self.agent.increment_agents(1);
        for _ in 0..num_steps {
            self.request_sender
                .as_ref()
                .expect("Call start before running steps.")
                .send(RequestMessage::Step())
                .expect("Can request transitions.");
            let transition = self
                .transition_receiver
                .recv()
                .expect("Can receive transitions.");
            items.push(transition.clone());

            if !self.eval {
                progress.items_processed += 1;
                processor.process_train(RLEvent::TimeStep(EvaluationItem::new(
                    transition.action_context,
                    progress.clone(),
                    None,
                )));

                if transition.done {
                    processor.process_train(RLEvent::EpisodeEnd(EvaluationItem::new(
                        EpisodeSummary {
                            episode_length: transition.ep_len,
                            cum_reward: transition.cum_reward,
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
        self.agent.decrement_agents(1);
        items
    }

    fn run_episodes(
        &mut self,
        num_episodes: usize,
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        _progress: &mut Progress,
    ) -> Vec<RLTrajectory<BT, RLC>> {
        let mut items = vec![];
        self.agent.increment_agents(1);
        for episode_num in 0..num_episodes {
            self.request_sender
                .as_ref()
                .expect("Call start before running episodes.")
                .send(RequestMessage::Episode())
                .expect("Can request episodes.");
            let trajectory = self
                .trajectory_receiver
                .recv()
                .expect("Main thread can receive trajectory.");

            for (i, step) in trajectory.timesteps.iter().enumerate() {
                if self.eval {
                    processor.process_valid(AgentEvaluationEvent::TimeStep(EvaluationItem::new(
                        step.action_context.clone(),
                        Progress::new(i, i),
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
                } else {
                    processor.process_train(RLEvent::TimeStep(EvaluationItem::new(
                        step.action_context.clone(),
                        Progress::new(i, i),
                        None,
                    )));

                    if step.done {
                        processor.process_train(RLEvent::EpisodeEnd(EvaluationItem::new(
                            EpisodeSummary {
                                episode_length: step.ep_len,
                                cum_reward: step.cum_reward,
                            },
                            Progress::new(episode_num + 1, num_episodes),
                            None,
                        )));
                    }
                }
            }

            items.push(trajectory);
            if interrupter.should_stop() {
                break;
            }
        }
        self.agent.decrement_agents(1);
        items
    }

    fn update_policy(&mut self, update: <RlPolicy<RLC> as Policy<RLC::Backend>>::PolicyState) {
        self.agent.update(update);
    }

    fn policy(
        &self,
    ) -> <RlPolicy<RLC> as Policy<<RLC as RLComponentsTypes>::Backend>>::PolicyState {
        self.agent.state()
    }
}

/// An asynchronous runner for multiple agent/environement interfaces.
pub struct MultiAgentEnvLoop<BT: Backend, RLC: RLComponentsTypes> {
    env_init: RLC::EnvInit,
    num_envs: usize,
    eval: bool,
    agent: AsyncPolicy<RLC::Backend, RLC::Policy>,
    deterministic: bool,
    device: Device<BT>,
    transition_receiver: Receiver<RLTimeStep<BT, RLC>>,
    transition_sender: Sender<RLTimeStep<BT, RLC>>,
    trajectory_receiver: Receiver<RLTrajectory<BT, RLC>>,
    trajectory_sender: Sender<RLTrajectory<BT, RLC>>,
    request_senders: Vec<Sender<RequestMessage>>,
}

impl<BT: Backend, RLC: RLComponentsTypes> MultiAgentEnvLoop<BT, RLC> {
    /// Create a new asynchronous runner for multiple agent/environement interfaces.
    pub fn new(
        env_init: RLC::EnvInit,
        num_envs: usize,
        eval: bool,
        agent: AsyncPolicy<RLC::Backend, RLC::Policy>,
        deterministic: bool,
        device: &Device<BT>,
    ) -> Self {
        let (transition_sender, transition_receiver) = std::sync::mpsc::channel();
        let (trajectory_sender, trajectory_receiver) = std::sync::mpsc::channel();
        Self {
            env_init,
            num_envs,
            eval,
            agent: agent.clone(),
            deterministic,
            device: device.clone(),
            transition_receiver,
            transition_sender,
            trajectory_receiver,
            trajectory_sender,
            request_senders: Vec::with_capacity(num_envs),
        }
    }
}

impl<BT, RLC> AgentEnvLoop<BT, RLC> for MultiAgentEnvLoop<BT, RLC>
where
    BT: Backend,
    RLC: RLComponentsTypes,
    RLC::Policy: Send + 'static,
    <RLC::Policy as Policy<RLC::Backend>>::PolicyState: Send,
    <RLC::Policy as Policy<RLC::Backend>>::ActionContext: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Observation: Send,
    <RLC::Policy as Policy<RLC::Backend>>::Action: Send,
    <RLC::Policy as Policy<RLC::Backend>>::ActionDistribution: Send,
{
    // TODO: start() shouldn't exist.
    fn start(&mut self) {
        // Double batching : The environments are always one step ahead of requests. This allows inference for the first batch of steps.
        self.agent.increment_agents(self.num_envs);

        for i in 0..self.num_envs {
            let mut runner = AgentEnvAsyncLoop::<BT, RLC>::new(
                self.env_init.clone(),
                i,
                self.eval,
                self.agent.clone(),
                self.deterministic,
                &self.device,
            );
            runner.transition_sender = self.transition_sender.clone();
            runner.trajectory_sender = self.trajectory_sender.clone();
            runner.start();
            self.request_senders
                .push(runner.request_sender.clone().unwrap());
        }

        // Double batching : The environments are always one step ahead.
        self.request_senders.iter().for_each(|s| {
            s.send(RequestMessage::Step())
                .expect("Main thread can send step requests.")
        });
    }

    fn run_steps(
        &mut self,
        num_steps: usize,
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<RLTimeStep<BT, RLC>> {
        let mut items = vec![];
        for _ in 0..num_steps {
            let transition = self
                .transition_receiver
                .recv()
                .expect("Can receive transitions.");
            items.push(transition.clone());

            self.request_senders[transition.env_id]
                .send(RequestMessage::Step())
                .expect("Main thread can request steps.");

            if !self.eval {
                progress.items_processed += 1;
                processor.process_train(RLEvent::TimeStep(EvaluationItem::new(
                    transition.action_context,
                    progress.clone(),
                    None,
                )));

                if transition.done {
                    processor.process_train(RLEvent::EpisodeEnd(EvaluationItem::new(
                        EpisodeSummary {
                            episode_length: transition.ep_len,
                            cum_reward: transition.cum_reward,
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
        processor: &mut RLEventProcessorType<RLC>,
        interrupter: &Interrupter,
        _progress: &mut Progress,
    ) -> Vec<RLTrajectory<BT, RLC>> {
        // Send `num_episodes` initial requests.
        let mut idx = vec![];
        if num_episodes < self.num_envs {
            let mut rng = rand::rng();
            let mut vec: Vec<usize> = (0..self.num_envs).collect();
            vec.shuffle(&mut rng);
            idx = vec.into_iter().take(num_episodes).collect();
        } else {
            idx = (0..self.num_envs).collect();
        }
        let num_requests = self.num_envs.min(num_episodes);
        idx.into_iter().for_each(|i| {
            self.request_senders[i]
                .send(RequestMessage::Episode())
                .expect("Main thread can request steps.");
        });

        let mut items = vec![];
        for episode_num in 0..num_episodes {
            let trajectory = self
                .trajectory_receiver
                .recv()
                .expect("Can receive trajectory.");
            items.push(trajectory.clone());
            if items.len() + num_requests <= num_episodes {
                self.request_senders[trajectory.timesteps[0].env_id]
                    .send(RequestMessage::Episode())
                    .expect("Main thread can request steps.");
            }
            for (i, step) in trajectory.timesteps.iter().enumerate() {
                if self.eval {
                    processor.process_valid(AgentEvaluationEvent::TimeStep(EvaluationItem::new(
                        step.action_context.clone(),
                        Progress::new(i, i),
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
                } else {
                    processor.process_train(RLEvent::TimeStep(EvaluationItem::new(
                        step.action_context.clone(),
                        Progress::new(i, i),
                        None,
                    )));

                    if step.done {
                        processor.process_train(RLEvent::EpisodeEnd(EvaluationItem::new(
                            EpisodeSummary {
                                episode_length: step.ep_len,
                                cum_reward: step.cum_reward,
                            },
                            Progress::new(episode_num + 1, num_episodes),
                            None,
                        )));
                    }
                }
            }

            if interrupter.should_stop() {
                break;
            }
        }

        items
    }

    fn policy(
        &self,
    ) -> <RlPolicy<RLC> as Policy<<RLC as RLComponentsTypes>::Backend>>::PolicyState {
        self.agent.state()
    }
}

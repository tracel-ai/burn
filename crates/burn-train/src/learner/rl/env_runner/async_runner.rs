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

/// Configuration for an async agent/environment loop.
pub struct AsyncAgentEnvLoopConfig {
    /// If the loop is used for evaluation (as opposed to training).
    pub eval: bool,
    /// If the agent should take action deterministically.
    pub deterministic: bool,
    /// An arbitrary ID for the loop.
    pub id: usize,
}

/// An asynchronous agent/environement interface.
pub struct AgentEnvAsyncLoop<BT: Backend, RLC: RLComponentsTypes> {
    eval: bool,
    agent: AsyncPolicy<RLC::Backend, RlPolicy<RLC>>,
    transition_receiver: Receiver<RLTimeStep<BT, RLC>>,
    trajectory_receiver: Receiver<RLTrajectory<BT, RLC>>,
    request_sender: Sender<RequestMessage>,
}

impl<BT: Backend, RLC: RLComponentsTypes> AgentEnvAsyncLoop<BT, RLC> {
    /// Create a new asynchronous runner.
    ///
    /// # Arguments
    ///
    /// * `env_init` - A function returning an environement instance.
    /// * `agent` - An [AsyncPolicy](AsyncPolicy) taking actions in the loop.
    /// * `config` - An [AsyncAgentEnvLoopConfig](AsyncAgentEnvLoopConfig).
    /// * `transition_sender` - Optional sender for transitions if you want to drive the requests from outside of the loop instance.
    /// * `trajectory_sender` - Optional sender for trajectories if you want to drive the requests from outside of the loop instance.
    ///
    /// # Returns
    ///
    /// An async Agent/Environement loop.
    pub fn new(
        env_init: RLC::EnvInit,
        agent: AsyncPolicy<RLC::Backend, RlPolicy<RLC>>,
        config: AsyncAgentEnvLoopConfig,
        transition_device: &Device<BT>,
        transition_sender: Option<Sender<RLTimeStep<BT, RLC>>>,
        trajectory_sender: Option<Sender<RLTrajectory<BT, RLC>>>,
    ) -> Self {
        let (loop_transition_sender, transition_receiver) = std::sync::mpsc::channel();
        let (loop_trajectory_sender, trajectory_receiver) = std::sync::mpsc::channel();
        let (request_sender, request_receiver) = std::sync::mpsc::channel();
        let loop_transition_sender = transition_sender.unwrap_or(loop_transition_sender);
        let loop_trajectory_sender = trajectory_sender.unwrap_or(loop_trajectory_sender);

        let device = transition_device.clone();
        let mut loop_agent = agent.clone();
        let eval = config.eval;

        let mut current_steps = vec![];
        let mut current_reward = 0.0;
        let mut step_num = 0;
        spawn(move || {
            let mut env = env_init.init();
            env.reset();

            let mut request_episode = false;
            loop {
                let state = env.state();
                let (action, context) =
                    loop_agent.action(state.clone().into(), config.deterministic);

                let env_action = RLC::Action::from(action);
                let step_result = env.step(env_action.clone());

                current_reward += step_result.reward;
                step_num += 1;

                let transition = Transition::new(
                    state.clone(),
                    step_result.next_state,
                    env_action,
                    Tensor::from_data([step_result.reward], &device),
                    Tensor::from_data(
                        [(step_result.done || step_result.truncated) as i32 as f64],
                        &device,
                    ),
                );

                if !request_episode {
                    loop_agent.decrement_agents(1);
                    let request = match request_receiver.recv() {
                        Ok(req) => req,
                        Err(err) => {
                            log::error!("Error in env runner : {}", err);
                            break;
                        }
                    };
                    loop_agent.increment_agents(1);

                    match request {
                        RequestMessage::Step() => (),
                        RequestMessage::Episode() => request_episode = true,
                    }
                }

                let time_step = TimeStep {
                    env_id: config.id,
                    transition,
                    done: step_result.done,
                    ep_len: step_num,
                    cum_reward: current_reward,
                    action_context: context[0].clone(),
                };
                current_steps.push(time_step.clone());

                if !request_episode && let Err(err) = loop_transition_sender.send(time_step) {
                    log::error!("Error in env runner : {}", err);
                    break;
                }

                if step_result.done || step_result.truncated {
                    if request_episode {
                        request_episode = false;
                        loop_trajectory_sender
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

        Self {
            eval,
            agent,
            transition_receiver,
            trajectory_receiver,
            request_sender,
        }
    }
}

impl<BT, RLC> AgentEnvLoop<BT, RLC> for AgentEnvAsyncLoop<BT, RLC>
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
        for _ in 0..num_steps {
            self.request_sender
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
                .send(RequestMessage::Episode())
                .expect("Can request episodes.");
            let trajectory = self
                .trajectory_receiver
                .recv()
                .expect("Main thread can receive trajectory.");

            for (i, step) in trajectory.timesteps.iter().enumerate() {
                // TODO : clean this.
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

    fn update_policy(&mut self, update: RLC::PolicyState) {
        self.agent.update(update);
    }

    fn policy(&self) -> RLC::PolicyState {
        self.agent.state()
    }
}

/// An asynchronous runner for multiple agent/environement interfaces.
pub struct MultiAgentEnvLoop<BT: Backend, RLC: RLComponentsTypes> {
    num_envs: usize,
    eval: bool,
    agent: AsyncPolicy<RLC::Backend, RLC::Policy>,
    transition_receiver: Receiver<RLTimeStep<BT, RLC>>,
    trajectory_receiver: Receiver<RLTrajectory<BT, RLC>>,
    request_senders: Vec<Sender<RequestMessage>>,
}

impl<BT: Backend, RLC: RLComponentsTypes> MultiAgentEnvLoop<BT, RLC> {
    /// Create a new asynchronous runner for multiple agent/environement interfaces.
    pub fn new(
        num_envs: usize,
        env_init: RLC::EnvInit,
        agent: AsyncPolicy<RLC::Backend, RLC::Policy>,
        eval: bool,
        deterministic: bool,
        device: &Device<BT>,
    ) -> Self {
        let (transition_sender, transition_receiver) = std::sync::mpsc::channel();
        let (trajectory_sender, trajectory_receiver) = std::sync::mpsc::channel();
        let mut request_senders = vec![];

        // Double batching : The environments are always one step ahead of requests. This allows inference for the first batch of steps.
        agent.increment_agents(num_envs);

        for i in 0..num_envs {
            let config = AsyncAgentEnvLoopConfig {
                eval,
                deterministic,
                id: i,
            };
            let runner = AgentEnvAsyncLoop::<BT, RLC>::new(
                env_init.clone(),
                agent.clone(),
                config,
                &device.clone(),
                Some(transition_sender.clone()),
                Some(trajectory_sender.clone()),
            );
            request_senders.push(runner.request_sender.clone());
        }

        // Double batching : The environments are always one step ahead.
        request_senders.iter().for_each(|s| {
            s.send(RequestMessage::Step())
                .expect("Main thread can send step requests.")
        });

        Self {
            num_envs,
            eval,
            agent: agent.clone(),
            transition_receiver,
            trajectory_receiver,
            request_senders,
        }
    }
}

impl<BT, RLC> AgentEnvLoop<BT, RLC> for MultiAgentEnvLoop<BT, RLC>
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

    fn update_policy(&mut self, update: RLC::PolicyState) {
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

    fn policy(&self) -> RLC::PolicyState {
        self.agent.state()
    }
}

#[cfg(test)]
mod tests {
    use burn_core::data::dataloader::Progress;
    use burn_rl::AsyncPolicy;

    use crate::learner::rl::env_runner::async_runner::AsyncAgentEnvLoopConfig;
    use crate::learner::rl::env_runner::base::AgentEnvLoop;
    use crate::learner::tests::{MockPolicyState, MockProcessor};
    use crate::{
        AgentEnvAsyncLoop, TestBackend,
        learner::tests::{MockEnvInit, MockPolicy, MockRLComponents},
    };
    use crate::{AsyncProcessorTraining, Interrupter, MultiAgentEnvLoop};

    fn setup_async_loop(
        state: usize,
        eval: bool,
        deterministic: bool,
    ) -> AgentEnvAsyncLoop<TestBackend, MockRLComponents> {
        let env_init = MockEnvInit;
        let agent = MockPolicy(state);
        let config = AsyncAgentEnvLoopConfig {
            eval,
            deterministic,
            id: 0,
        };
        AgentEnvAsyncLoop::<TestBackend, MockRLComponents>::new(
            env_init,
            AsyncPolicy::new(1, agent),
            config,
            &Default::default(),
            None,
            None,
        )
    }

    fn setup_multi_loop(
        num_envs: usize,
        autobatch_size: usize,
        state: usize,
        eval: bool,
        deterministic: bool,
    ) -> MultiAgentEnvLoop<TestBackend, MockRLComponents> {
        let env_init = MockEnvInit;
        let agent = MockPolicy(state);
        MultiAgentEnvLoop::<TestBackend, MockRLComponents>::new(
            num_envs,
            env_init,
            AsyncPolicy::new(autobatch_size, agent),
            eval,
            deterministic,
            &Default::default(),
        )
    }

    #[test]
    fn test_policy_async_loop() {
        let runner = setup_async_loop(1000, false, false);
        let policy_state = runner.policy();
        assert_eq!(policy_state.0, 1000);
    }

    #[test]
    fn test_update_policy_async_loop() {
        let mut runner = setup_async_loop(0, false, false);

        runner.update_policy(MockPolicyState(1));
        assert_eq!(runner.policy().0, 1);
    }

    #[test]
    fn run_steps_returns_requested_number_async_loop() {
        let mut runner = setup_async_loop(0, false, false);
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
    fn run_episodes_returns_requested_number_async_loop() {
        let mut runner = setup_async_loop(0, false, false);
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

    #[test]
    fn test_policy_multi_loop() {
        let runner = setup_multi_loop(4, 4, 1000, false, false);
        let policy_state = runner.policy();
        assert_eq!(policy_state.0, 1000);
    }

    #[test]
    fn test_update_policy_multi_loop() {
        let mut runner = setup_multi_loop(4, 4, 0, false, false);

        runner.update_policy(MockPolicyState(1));
        assert_eq!(runner.policy().0, 1);
    }

    #[test]
    fn run_steps_returns_requested_number_multi_loop() {
        fn run_test(num_envs: usize, autobatch_size: usize) {
            let mut runner = setup_multi_loop(num_envs, autobatch_size, 0, false, false);
            let mut processor = AsyncProcessorTraining::new(MockProcessor);
            let mut interrupter = Interrupter::new();
            let mut progress = Progress {
                items_processed: 0,
                items_total: 1,
            };

            // Kickstart tests by running some steps to make sure it's not a double batching edge case success.
            let steps = runner.run_steps(8, &mut processor, &mut interrupter, &mut progress);
            assert_eq!(steps.len(), 8);

            for i in 0..16 {
                let steps = runner.run_steps(i, &mut processor, &mut interrupter, &mut progress);
                assert_eq!(steps.len(), i);
            }
        }

        // num_envs == autobatch_size
        run_test(1, 1);
        run_test(4, 4);
        // num_envs < autobatch_size
        run_test(1, 2);
        run_test(1, 3);
        run_test(2, 3);
        run_test(2, 4);
        run_test(5, 19);
        // num_envs > autobatch_size
        run_test(2, 1);
        run_test(8, 1);
        run_test(3, 2);
        run_test(8, 2);
        run_test(8, 3);
        run_test(8, 7);
    }

    #[test]
    fn run_episodes_returns_requested_number_multi_loop() {
        fn run_test(num_envs: usize, autobatch_size: usize) {
            let mut runner = setup_multi_loop(num_envs, autobatch_size, 0, false, false);
            let mut processor = AsyncProcessorTraining::new(MockProcessor);
            let mut interrupter = Interrupter::new();
            let mut progress = Progress {
                items_processed: 0,
                items_total: 1,
            };

            // Kickstart tests by running some episodes to make sure it's not a double batching edge case success.
            let trajectories =
                runner.run_episodes(8, &mut processor, &mut interrupter, &mut progress);
            assert_eq!(trajectories.len(), 8);
            for j in 0..8 {
                assert_ne!(trajectories[j].timesteps.len(), 0);
            }

            for i in 0..16 {
                let trajectories =
                    runner.run_episodes(i, &mut processor, &mut interrupter, &mut progress);
                assert_eq!(trajectories.len(), i);
                for j in 0..i {
                    assert_ne!(trajectories[j].timesteps.len(), 0);
                }
            }
        }

        // num_envs == autobatch_size
        run_test(1, 1);
        run_test(4, 4);
        // num_envs < autobatch_size
        run_test(1, 2);
        run_test(1, 3);
        run_test(2, 3);
        run_test(2, 4);
        run_test(5, 19);
        // num_envs > autobatch_size
        run_test(2, 1);
        run_test(8, 1);
        run_test(3, 2);
        run_test(8, 2);
        run_test(8, 3);
        run_test(8, 7);
    }
}

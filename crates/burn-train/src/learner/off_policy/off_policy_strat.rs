use crate::{
    AsyncEnvArrayRunner, AsyncEnvRunner, EnvRunner, EvaluationItem, EventProcessorTraining,
    RLComponents, RLEvent, RLEventProcessorType, ReinforcementLearningComponentsTypes,
    ReinforcementLearningStrategy,
};
use burn_core::{self as burn};
use burn_core::{config::Config, data::dataloader::Progress, tensor::Device};
use burn_rl::{AgentLearner, AsyncPolicy, Policy, TransitionBuffer};

/// Parameters of an on policy training with multi environments and double-batching.
#[derive(Config, Debug)]
pub struct OffPolicyConfig {
    /// The number of environments to run simultaneously for experience collection.
    #[config(default = 1)]
    pub num_envs: usize,
    /// Number of environment state to accumulate before running one step of inference with the policy.
    /// Must be equal or less than the number of simultaneous environments.
    #[config(default = 1)]
    pub autobatch_size: usize,
    /// Max number of transitions stored in the replay buffer.
    #[config(default = 1024)]
    pub replay_buffer_size: usize,
    /// The number of steps to collect between each step of training.
    #[config(default = 1)]
    pub train_interval: usize,
    /// The number of steps to collect between each evaluation.
    #[config(default = 10_000)]
    pub eval_interval: usize,
    /// The number of episodes to run for each evaluation.
    #[config(default = 1)]
    pub eval_episodes: usize,
    /// The number of transition to train on.
    #[config(default = 32)]
    pub train_batch_size: usize,
}

/// Off-policy reinforcement learning strategy with multi-env experience collection and double-batching.
pub struct OffPolicyStrategy<RLC: ReinforcementLearningComponentsTypes> {
    device: Device<RLC::Backend>,
    config: OffPolicyConfig,
}
impl<RLC: ReinforcementLearningComponentsTypes> OffPolicyStrategy<RLC> {
    /// Create a new off-policy base strategy.
    pub fn new(device: Device<RLC::Backend>) -> Self {
        Self {
            device,
            config: OffPolicyConfig::new(),
        }
    }

    /// Update the default config.
    pub fn with_config(self, config: OffPolicyConfig) -> Self {
        Self {
            device: self.device,
            config,
        }
    }
}

impl<RLC> ReinforcementLearningStrategy<RLC> for OffPolicyStrategy<RLC>
where
    RLC: ReinforcementLearningComponentsTypes,
{
    fn fit(
        &self,
        training_components: RLComponents<RLC>,
        learner_agent: &mut RLC::LearningAgent,
        starting_epoch: usize,
    ) -> (RLC::Policy, RLEventProcessorType<RLC>) {
        let mut event_processor = training_components.event_processor;
        let mut checkpointer = training_components.checkpointer;
        let num_steps_total = training_components.num_steps;

        let mut env_runner = AsyncEnvArrayRunner::<RLC::Backend, RLC>::new(
            self.config.num_envs,
            false,
            AsyncPolicy::new(
                self.config.num_envs.min(self.config.autobatch_size),
                learner_agent.policy(),
            ),
            false,
            &self.device,
        );
        env_runner.start();
        let mut env_runner_valid = AsyncEnvRunner::<RLC::Backend, RLC>::new(
            0,
            true,
            AsyncPolicy::new(1, learner_agent.policy()),
            true,
            &self.device,
        );
        env_runner_valid.start();
        let mut transition_buffer = TransitionBuffer::<
            <RLC::LearningAgent as AgentLearner<RLC::Backend>>::TrainingInput,
        >::new(self.config.replay_buffer_size);

        let mut valid_next = self.config.eval_interval + starting_epoch - 1;
        let mut progress = Progress {
            items_processed: starting_epoch,
            items_total: num_steps_total,
        };

        let mut intermediary_update: Option<<RLC::Policy as Policy<RLC::Backend>>::PolicyState> =
            None;
        while progress.items_processed < num_steps_total {
            if training_components.interrupter.should_stop() {
                let reason = training_components
                    .interrupter
                    .get_message()
                    .unwrap_or(String::from("Reason unknown"));
                log::info!("Training interrupted: {reason}");
                break;
            }

            let previous_steps = progress.items_processed;
            let items = env_runner.run_steps(
                self.config.train_interval,
                false,
                &mut event_processor,
                &training_components.interrupter,
                &mut progress,
            );

            transition_buffer
                .append(&mut items.iter().map(|i| i.transition.clone().into()).collect());

            if transition_buffer.len() >= self.config.train_batch_size {
                if let Some(u) = intermediary_update {
                    env_runner.update_policy(u.clone());
                }
                let transitions = transition_buffer.random_sample(self.config.train_batch_size);
                let train_item = learner_agent.train(transitions);
                intermediary_update = Some(train_item.policy);

                event_processor.process_train(RLEvent::TrainStep(EvaluationItem::new(
                    train_item.item,
                    progress.clone(),
                    None,
                )));
            }

            if valid_next > previous_steps && valid_next <= progress.items_processed {
                env_runner_valid.update_policy(learner_agent.policy().state());
                env_runner_valid.run_episodes(
                    self.config.eval_episodes,
                    true,
                    &mut event_processor,
                    &training_components.interrupter,
                    &mut progress,
                );

                if let Some(checkpointer) = &mut checkpointer {
                    checkpointer.checkpoint(
                        &env_runner.policy(),
                        &learner_agent,
                        valid_next,
                        &training_components.event_store,
                    );
                }

                valid_next += self.config.eval_interval;
            }
        }

        (learner_agent.policy(), event_processor)
    }
}

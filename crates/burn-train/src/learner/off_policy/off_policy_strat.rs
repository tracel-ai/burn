use crate::{
    AsyncEnvArrayRunner, AsyncEnvRunner, EnvRunner, EvaluationItem, EventProcessorTraining,
    RLComponents, RLEvent, RLEventProcessorType, ReinforcementLearningComponentsTypes,
    ReinforcementLearningStrategy,
};
use burn_core::{data::dataloader::Progress, tensor::Device};
use burn_rl::{AgentLearner, AsyncPolicy, Policy, TransitionBuffer};

/// Base strategy for off-policy reinforcement learning.
pub struct SimpleOffPolicyStrategy<RLC: ReinforcementLearningComponentsTypes> {
    device: Device<RLC::Backend>,
}
impl<RLC: ReinforcementLearningComponentsTypes> SimpleOffPolicyStrategy<RLC> {
    /// Create a new off-policy base strategy.
    pub fn new(device: Device<RLC::Backend>) -> Self {
        Self { device }
    }
}

impl<RLC> ReinforcementLearningStrategy<RLC> for SimpleOffPolicyStrategy<RLC>
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

        struct MultiEnvConfig {
            num_envs: usize,
            autobatch_size: usize,
        }
        let multi_env_config = MultiEnvConfig {
            num_envs: 16,
            autobatch_size: 4,
        };
        // TODO: pq on a besoin du type?
        let mut env_runner = AsyncEnvArrayRunner::<RLC::Backend, RLC>::new(
            multi_env_config.num_envs,
            false,
            AsyncPolicy::new(multi_env_config.autobatch_size, learner_agent.policy()),
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
        >::new(2048);

        let train_interval = 8;
        let valid_interval = 5000;
        let valid_episodes = 5;
        let mut valid_next = valid_interval + starting_epoch - 1;
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
                train_interval,
                false,
                &mut event_processor,
                &training_components.interrupter,
                &mut progress,
            );

            transition_buffer
                .append(&mut items.iter().map(|i| i.transition.clone().into()).collect());

            let batch_size = 128;
            if transition_buffer.len() >= batch_size {
                if let Some(u) = intermediary_update {
                    env_runner.update_policy(u.clone());
                }
                let train_item = learner_agent.train(&transition_buffer);
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
                    valid_episodes,
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

                valid_next += valid_interval;
            }
        }

        (learner_agent.policy(), event_processor)
    }
}

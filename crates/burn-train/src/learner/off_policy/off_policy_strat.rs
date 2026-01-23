use crate::{
    AsyncEnvArrayRunner, AsyncEnvRunner, EnvRunner, EpisodeSummary, EvaluationItem,
    EventProcessorTraining, RLComponents, RLEvent, RLEventProcessorType,
    ReinforcementLearningComponentsTypes, ReinforcementLearningStrategy,
};
use burn_core::{data::dataloader::Progress, tensor::Device};
use burn_rl::{AsyncPolicy, LearnerAgent, Policy, TransitionBuffer};

/// Base strategy for off-policy reinforcement learning.
pub struct SimpleOffPolicyStrategy<OC: ReinforcementLearningComponentsTypes> {
    device: Device<OC::Backend>,
}
impl<OC: ReinforcementLearningComponentsTypes> SimpleOffPolicyStrategy<OC> {
    /// Create a new off-policy base strategy.
    pub fn new(device: Device<OC::Backend>) -> Self {
        Self { device }
    }
}

impl<OC> ReinforcementLearningStrategy<OC> for SimpleOffPolicyStrategy<OC>
where
    OC: ReinforcementLearningComponentsTypes,
{
    fn fit(
        &self,
        training_components: RLComponents<OC>,
        learner_agent: &mut OC::LearningAgent,
        starting_epoch: usize,
    ) -> (OC::Policy, RLEventProcessorType<OC>) {
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
        let mut env_runner = AsyncEnvArrayRunner::<OC::Backend, OC>::new(
            multi_env_config.num_envs,
            AsyncPolicy::new(multi_env_config.autobatch_size, learner_agent.policy()),
            false,
            &self.device,
        );
        env_runner.start();
        let mut env_runner_valid = AsyncEnvRunner::<OC::Backend, OC>::new(
            0,
            AsyncPolicy::new(1, learner_agent.policy()),
            true,
            &self.device,
        );
        env_runner_valid.start();
        let mut transition_buffer = TransitionBuffer::<
            <OC::LearningAgent as LearnerAgent<OC::Backend>>::TrainingInput,
        >::new(2048);

        let train_interval = 8;
        let valid_interval = 5000;
        let valid_episodes = 5;
        let mut valid_next = false;
        let mut num_steps = starting_epoch;
        while num_steps < num_steps_total {
            if training_components.interrupter.should_stop() {
                let reason = training_components
                    .interrupter
                    .get_message()
                    .unwrap_or(String::from("Reason unknown"));
                log::info!("Training interrupted: {reason}");
                break;
            }

            let items = env_runner.run_steps(
                train_interval,
                false,
                &mut event_processor,
                &training_components.interrupter,
            );

            for item in items {
                // TODO : For now, every env returns a transition and that is what is stored in the replay buffer.
                // Maybe the agent should define what is stored from the transition (LearnerAgent::TrainingInput).
                transition_buffer.push(item.transition.into());

                num_steps += 1;
                event_processor.process_train(RLEvent::TimeStep(EvaluationItem::new(
                    item.action_context,
                    Progress {
                        items_processed: num_steps,
                        items_total: num_steps_total,
                    },
                    None,
                )));

                if num_steps % valid_interval == 0 {
                    valid_next = true;
                }

                if item.done {
                    event_processor.process_train(RLEvent::EpisodeEnd(EvaluationItem::new(
                        EpisodeSummary {
                            episode_length: item.ep_len,
                            cum_reward: item.cum_reward,
                        },
                        Progress {
                            items_processed: num_steps,
                            items_total: num_steps_total,
                        },
                        None,
                    )));
                }
            }

            let batch_size = 128;
            if transition_buffer.len() >= batch_size {
                // let batch = transition_buffer.random_sample(batch_size);
                // if let Some(ref u) = intermediary_update {
                //     self.runner.update_policy(u.clone());
                // }
                // intermediary_update = Some(self.agent.train(batch));

                let update = learner_agent.train(&transition_buffer);
                env_runner.update_policy(update.policy);

                event_processor.process_train(RLEvent::TrainStep(EvaluationItem::new(
                    update.item,
                    Progress {
                        items_processed: num_steps,
                        items_total: num_steps_total,
                    },
                    None,
                )));
            }

            if valid_next {
                valid_next = false;

                env_runner_valid.update_policy(learner_agent.policy().state());
                env_runner_valid.run_episodes(
                    valid_episodes,
                    true,
                    &mut event_processor,
                    &training_components.interrupter,
                );

                if let Some(checkpointer) = &mut checkpointer {
                    checkpointer.checkpoint(
                        &learner_agent.policy(),
                        &learner_agent,
                        num_steps,
                        &training_components.event_store,
                    );
                }
            }
        }

        (learner_agent.policy(), event_processor)
    }
}

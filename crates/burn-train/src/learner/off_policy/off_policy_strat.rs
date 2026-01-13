use crate::{
    AsyncEnvArrayRunner, AsyncEnvRunner, EnvRunner, EpisodeSummary, EventProcessorTraining,
    LearnerItem, OffPolicyLearningComponentsTypes, RLComponents, ReinforcementLearningStrategy,
    RlEventProcessor,
};
use burn_core::{data::dataloader::Progress, tensor::Device};
use burn_rl::{Agent, AsyncAgent, LearnerAgent, TransitionBuffer};

/// Simplest learning strategy possible, with only a single devices doing both the training and
/// validation.
pub struct SimpleOffPolicyStrategy<OC: OffPolicyLearningComponentsTypes> {
    // device: Device<TrainingBackend<OC::LC>>,
    device: Device<OC::Backend>,
}
impl<OC: OffPolicyLearningComponentsTypes> SimpleOffPolicyStrategy<OC> {
    // pub fn new(device: Device<TrainingBackend<OC::LC>>) -> Self {
    pub fn new(device: Device<OC::Backend>) -> Self {
        Self { device }
    }
}

impl<OC: OffPolicyLearningComponentsTypes> ReinforcementLearningStrategy<OC>
    for SimpleOffPolicyStrategy<OC>
{
    fn fit(
        &self,
        training_components: RLComponents<OC>,
        learner_agent: &mut OC::LearningAgent,
        starting_epoch: usize,
    ) -> (
        // <OC::LearningAgent as Agent<TrainingBackend<OC::LC>, OC::Env>>::Policy,
        <OC::LearningAgent as Agent<OC::Backend, OC::Env>>::Policy,
        RlEventProcessor<OC>,
    ) {
        let mut event_processor = training_components.event_processor;
        let mut checkpointer = training_components.checkpointer;
        let mut early_stopping = training_components.early_stopping;
        let num_epochs = training_components.num_epochs;

        struct MultiEnvConfig {
            num_envs: usize,
            autobatch_size: usize,
        }
        let multi_env_config = MultiEnvConfig {
            num_envs: 1,
            autobatch_size: 1,
        };
        // TODO: pq on a besoin du type?
        let mut env_runner = AsyncEnvArrayRunner::<OC::Backend, OC>::new(
            multi_env_config.num_envs,
            AsyncAgent::new(multi_env_config.autobatch_size, learner_agent.clone()),
            false,
            &self.device,
        );
        env_runner.start();
        let mut env_runner_valid = AsyncEnvRunner::<OC::Backend, OC>::new(
            0,
            AsyncAgent::new(1, learner_agent.clone()),
            true,
            &self.device,
        );
        env_runner_valid.start();
        let mut transition_buffer = TransitionBuffer::<
            <OC::LearningAgent as LearnerAgent<OC::Backend, OC::Env>>::TrainingInput,
        >::new(2048);

        let num_steps = 8;
        let mut num_items = 0;
        let valid_interval = 100;
        let valid_episodes = 10;

        let mut global_valid_iteration = 0;
        for epoch in starting_epoch..training_components.num_epochs + 1 {
            if training_components.interrupter.should_stop() {
                let reason = training_components
                    .interrupter
                    .get_message()
                    .unwrap_or(String::from("Reason unknown"));
                log::info!("Training interrupted: {reason}");
                break;
            }

            let items = env_runner.run_steps(
                num_steps,
                false,
                &mut event_processor,
                &training_components.interrupter,
            );

            for item in items {
                // TODO : For now, every env returns a transition and that is what is stored in the replay buffer.
                // Maybe the agent should define what is stored from the transition (LearnerAgent::TrainingInput).
                transition_buffer.push(item.transition.into());

                num_items += 1;
                event_processor.process_train(crate::LearnerEvent::ProcessedItem(
                    LearnerItem::new(
                        RlItemTypesTrain::Step(item.action_context),
                        Progress::new(epoch, training_components.num_epochs),
                        epoch,
                        training_components.num_epochs,
                        num_items,
                        None,
                    ),
                ));

                if item.done {
                    event_processor.process_train(crate::LearnerEvent::ProcessedItem(
                        LearnerItem::new(
                            RlItemTypesTrain::EpisodeSummary(EpisodeSummary {
                                episode_length: item.ep_len,
                                total_reward: item.cum_reward,
                            }),
                            Progress::new(epoch, training_components.num_epochs),
                            epoch,
                            training_components.num_epochs,
                            epoch,
                            None,
                        ),
                    ));
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

                event_processor.process_train(crate::LearnerEvent::ProcessedItem(
                    LearnerItem::new(
                        RlItemTypesTrain::TrainStep(update.item),
                        Progress::new(epoch, training_components.num_epochs),
                        epoch,
                        training_components.num_epochs,
                        epoch,
                        None,
                    ),
                ));
            }

            // if let Some(checkpointer) = &mut checkpointer {
            //     checkpointer.checkpoint(&learner, epoch, &training_components.event_store);
            // }

            if epoch % valid_interval == 0 {
                env_runner_valid.update_policy(learner_agent.policy());
                global_valid_iteration += 1;
                env_runner_valid.run_episodes(
                    valid_episodes,
                    true,
                    &mut event_processor,
                    &training_components.interrupter,
                    global_valid_iteration,
                    training_components.num_epochs,
                );
                // event_processor.process_valid(LearnerEvent::EndEpoch(global_valid_iteration));
            }

            if let Some(early_stopping) = &mut early_stopping
                && early_stopping.should_stop(epoch, &training_components.event_store)
            {
                break;
            }
        }

        (learner_agent.policy(), event_processor)
    }
}

use burn::{
    grad_clipping::GradientClippingConfig,
    optim::AdamWConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        OffPolicyConfig, RLTraining,
        metric::{CumulativeRewardMetric, EpisodeLengthMetric, ExplorationRateMetric, LossMetric},
    },
};

use crate::{
    agent::{DqnAgentConfig, DqnLearningAgent, MlpNet, MlpNetConfig},
    env::CartPoleWrapper,
};

static ARTIFACT_DIR: &str = "/tmp/burn-example-dqn-agent";

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let dqn_config = DqnAgentConfig {
        gamma: 0.99,
        learning_rate: 3e-4,
        tau: 0.005,
        epsilon_start: 0.9,
        epsilon_end: 0.01,
        epsilon_decay: 4000.0,
    };
    let model_config = MlpNetConfig {
        num_layers: 3,
        dropout: 0.0,
        d_input: 4,
        d_output: 2,
        d_hidden: 128,
    };
    let learning_config = OffPolicyConfig {
        num_envs: 16,
        autobatch_size: 8,
        replay_buffer_size: 2048,
        train_interval: 8,
        eval_interval: 10_000,
        eval_episodes: 5,
        train_batch_size: 128,
    };
    let policy_model = MlpNet::<B>::new(&model_config, &device);
    let optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(10.0)))
        .init();
    let agent = DqnLearningAgent::new(policy_model, optimizer, dqn_config);
    let learner = RLTraining::new(ARTIFACT_DIR, CartPoleWrapper::new)
        .metrics_train_step((LossMetric::new(),))
        .metrics_env_step((ExplorationRateMetric::new(),))
        .metrics_episode((EpisodeLengthMetric::new(), CumulativeRewardMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_steps(40_000)
        // .checkpoint(10_000)
        .with_learning_strategy(burn::train::RLStrategies::OffPolicyStrategy((
            device,
            learning_config,
        )))
        .summary();

    let _policy = learner.launch(agent);
}

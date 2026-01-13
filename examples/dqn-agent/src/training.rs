use burn::{
    grad_clipping::GradientClippingConfig,
    optim::AdamWConfig,
    tensor::backend::AutodiffBackend,
    train::{
        MetricEarlyStoppingStrategy, OffPolicyLearning, StoppingCondition,
        metric::{
            AccuracyMetric, EpisodeLengthMetric, ExplorationRateInput, ExplorationRateMetric,
            LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};

use crate::{
    agent::{DqnAgentConfig, DqnLearningAgent, MlpNet, MlpNetConfig},
    env::CartPoleWrapper,
    utils::EpsilonGreedyPolicy,
};

static ARTIFACT_DIR: &str = "/tmp/burn-example-dqn-agent";

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let dqn_config = DqnAgentConfig {
        gamma: 0.99,
        learning_rate: 3e-4,
        tau: 0.005,
        epsilon_start: 0.9,
        epsilon_end: 0.01,
        epsilon_decay: 2500.0,
    };
    let model_config = MlpNetConfig {
        num_layers: 3,
        dropout: 0.0,
        d_input: 4,
        d_output: 2,
        d_hidden: 128,
    };
    let policy_model = MlpNet::<B>::new(&model_config, &device);
    let optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(10.0)))
        .init();
    let exploration_policy = EpsilonGreedyPolicy::new(
        dqn_config.epsilon_start,
        dqn_config.epsilon_end,
        dqn_config.epsilon_decay,
    );
    let agent = DqnLearningAgent::<_, CartPoleWrapper, _, _>::new(
        policy_model,
        optimizer,
        dqn_config,
        exploration_policy,
        &device,
    );
    let learner = OffPolicyLearning::new(ARTIFACT_DIR)
        .metrics((
            EpisodeLengthMetric::new(),
            ExplorationRateMetric::new(),
            AccuracyMetric::new(),
        ))
        .metric_train_numeric(LossMetric::new())
        // .metric_train_numeric(LearningRateMetric::new())
        // .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &EpisodeLengthMetric::new(),
            Aggregate::Mean,
            Direction::Highest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 5 },
        ))
        .num_episodes(5000)
        .summary();

    let policy = learner.launch(agent);
}

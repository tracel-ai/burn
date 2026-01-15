use burn::{
    grad_clipping::GradientClippingConfig,
    optim::AdamWConfig,
    tensor::backend::AutodiffBackend,
    train::{
        MetricEarlyStoppingStrategy, OffPolicyLearning, ReinforcementLearningComponentsMarker,
        StoppingCondition,
        metric::{
            EpisodeLengthMetric, ExplorationRateMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
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
    let agent = DqnLearningAgent::<_, CartPoleWrapper, _, _>::new(
        policy_model,
        optimizer,
        dqn_config,
        &device,
    );
    let learner = OffPolicyLearning::<
        ReinforcementLearningComponentsMarker<_, CartPoleWrapper, _, _, _>,
    >::new(ARTIFACT_DIR)
    .train_step_metrics((LossMetric::new(),))
    .env_step_metrics((ExplorationRateMetric::new(),))
    .episode_metrics((EpisodeLengthMetric::new(),))
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

    let _policy = learner.launch(agent);
    // /// TODO:
    // let _policy = learner.launch(agent, CartPoleWrapper::new);
}

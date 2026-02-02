use std::marker::PhantomData;

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::Record;
use burn::rl::{Batchable, Policy, PolicyLearner, PolicyState, RLTrainOutput, TransitionBatch};
use burn::tensor::Transaction;
use burn::tensor::activation::softmax;
use burn::train::ItemLazy;
use burn::train::metric::{Adaptor, LossInput};
use burn::{
    Tensor,
    config::Config,
    module::AutodiffModule,
    nn::{self, loss::MseLoss},
    optim::{GradientsParams, Optimizer},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rng;

use crate::utils::{
    EpsilonGreedyPolicy, EpsilonGreedyPolicyState, create_lin_layers, soft_update_linear,
};

pub trait DiscreteActionModel<B: Backend>: Module<B> {
    type Input: Clone + Send + Batchable;

    fn forward(&self, input: Self::Input) -> DiscreteLogitsTensor<B, 2>;
}

#[derive(Config, Debug)]
pub struct MlpNetConfig {
    /// The number of layers.
    #[config(default = 3)]
    pub num_layers: usize,
    /// The dropout rate.
    #[config(default = 0.)]
    pub dropout: f64,
    /// The input dimension.
    #[config(default = 4)]
    pub d_input: usize,
    /// The output dimension.
    #[config(default = 2)]
    pub d_output: usize,
    /// The size of hidden layers.
    #[config(default = 256)]
    pub d_hidden: usize,
}

/// Multilayer Perceptron Network.
#[derive(Module, Debug)]
pub struct MlpNet<B: Backend> {
    pub linears: Vec<nn::Linear<B>>,
    pub dropout: nn::Dropout,
    pub activation: nn::Relu,
}

impl<B: Backend> MlpNet<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &MlpNetConfig, device: &B::Device) -> Self {
        Self {
            linears: create_lin_layers(
                config.num_layers,
                config.d_input,
                config.d_hidden,
                config.d_output,
                device,
            ),
            dropout: nn::DropoutConfig::new(config.dropout).init(),
            activation: nn::Relu::new(),
        }
    }
}

#[derive(Clone)]
pub struct ObservationTensor<B: Backend, const D: usize> {
    pub state: Tensor<B, D>,
}

impl<B: Backend, const D: usize> Batchable for ObservationTensor<B, D> {
    fn batch(value: Vec<Self>) -> Self {
        let tensors = value.iter().map(|v| v.state.clone()).collect();
        Self {
            state: Tensor::cat(tensors, 0),
        }
    }

    fn unbatch(self) -> Vec<Self> {
        self.state
            .split(1, 0)
            .iter()
            .map(|s| ObservationTensor { state: s.clone() })
            .collect()
    }
}

impl<B: Backend> DiscreteActionModel<B> for MlpNet<B> {
    type Input = ObservationTensor<B, 2>;

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, d_input]`
    /// - output: `[batch_size, d_output]`
    fn forward(&self, input: Self::Input) -> DiscreteLogitsTensor<B, 2> {
        let mut x = input.state;

        for (i, linear) in self.linears.iter().enumerate() {
            x = linear.forward(x);
            x = self.dropout.forward(x);
            if i < self.linears.len() - 1 {
                x = self.activation.forward(x);
            }
        }

        DiscreteLogitsTensor { logits: x }
    }
}

#[derive(Config, Debug)]
pub struct DqnAgentConfig {
    /// Discount factor (How to value long-term vs short-term rewards)
    #[config(default = 0.99)]
    pub gamma: f64,
    /// The learning rate
    #[config(default = 3e-4)]
    pub learning_rate: f64,
    /// The soft update rate of the target network
    #[config(default = 0.005)]
    pub tau: f64,
    /// Initial value of epsilon (Probability to choose a random action)
    #[config(default = 0.9)]
    pub epsilon_start: f64,
    /// Final value of epsilon (Probability to choose a random action)
    #[config(default = 0.01)]
    pub epsilon_end: f64,
    /// The exponential rate at which the epsilon value decays. Higher = slower decay
    #[config(default = 2500.0)]
    pub epsilon_decay: f64,
}

pub trait TargetModel<B: Backend> {
    fn soft_update(&self, that: &Self, tau: f64) -> Self;
}

impl<B: Backend> TargetModel<B> for MlpNet<B> {
    fn soft_update(&self, that: &Self, tau: f64) -> Self {
        let mut linears = Vec::with_capacity(self.linears.len());
        for i in 0..self.linears.len() {
            let layer = soft_update_linear(self.linears[i].clone(), &that.linears[i].clone(), tau);
            linears.insert(i, layer);
        }
        Self {
            linears,
            dropout: self.dropout.clone(),
            activation: self.activation.clone(),
        }
    }
}

#[derive(Clone)]
pub struct DqnState<B: Backend, M: DiscreteActionModel<B>> {
    model: M,
    _backend: PhantomData<B>,
}

impl<B: Backend, M: DiscreteActionModel<B>> PolicyState<B> for DqnState<B, M> {
    type Record = M::Record;

    fn into_record(self) -> Self::Record {
        self.model.clone().into_record()
    }

    fn load_record(&self, record: Self::Record) -> Self {
        Self {
            model: self.model.clone().load_record(record),
            _backend: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct DQN<B: Backend, M: DiscreteActionModel<B>> {
    model: M,
    _backend: PhantomData<B>,
}

impl<B: Backend, M: DiscreteActionModel<B>> DQN<B, M> {
    pub fn new(policy: M) -> Self {
        Self {
            model: policy,
            _backend: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct DiscreteLogitsTensor<B: Backend, const D: usize> {
    pub logits: Tensor<B, D>,
}

impl<B: Backend, const D: usize> Batchable for DiscreteLogitsTensor<B, D> {
    fn batch(value: Vec<Self>) -> Self {
        let tensors = value.iter().map(|v| v.logits.clone()).collect();
        Self {
            logits: Tensor::cat(tensors, 0),
        }
    }

    fn unbatch(self) -> Vec<Self> {
        self.logits
            .split(1, 0)
            .iter()
            .map(|l| DiscreteLogitsTensor { logits: l.clone() })
            .collect()
    }
}

#[derive(Clone)]
pub struct DiscreteActionTensor<B: Backend, const D: usize> {
    pub actions: Tensor<B, D>,
}

impl<B: Backend, const D: usize> Batchable for DiscreteActionTensor<B, D> {
    fn batch(value: Vec<Self>) -> Self {
        let tensors = value.iter().map(|v| v.actions.clone()).collect();
        Self {
            actions: Tensor::cat(tensors, 0),
        }
    }

    fn unbatch(self) -> Vec<Self> {
        self.actions
            .split(1, 0)
            .iter()
            .map(|a| DiscreteActionTensor { actions: a.clone() })
            .collect()
    }
}

impl<B: Backend, M: DiscreteActionModel<B>> Policy<B> for DQN<B, M> {
    type Observation = M::Input;
    type ActionDistribution = DiscreteLogitsTensor<B, 2>;
    type Action = DiscreteActionTensor<B, 2>;

    type ActionContext = ();
    type PolicyState = DqnState<B, M>;

    fn forward(&mut self, states: Self::Observation) -> Self::ActionDistribution {
        self.model.forward(states)
    }

    fn action(
        &mut self,
        states: Self::Observation,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>) {
        let logits = self.forward(states).logits;
        if deterministic {
            let output = DiscreteActionTensor {
                actions: logits.argmax(1).float(),
            };
            return (output, vec![]);
        }

        let mut actions = vec![];
        let probs = softmax(logits, 1);
        let probs = probs.split(1, 0);
        let mut rng = rng();
        for i in 0..probs.len() {
            let dist = WeightedIndex::new(probs[i].to_data().to_vec::<f32>().unwrap()).unwrap();
            let action = dist.sample(&mut rng);
            actions.push(Tensor::<B, 1>::from_floats([action], &probs[i].device()));
        }

        let output = DiscreteActionTensor {
            actions: Tensor::stack(actions, 1),
        };
        return (output, vec![]);
    }

    fn update(&mut self, update: Self::PolicyState) {
        self.model = update.model;
    }

    fn state(&self) -> Self::PolicyState {
        DqnState {
            model: self.model.clone(),
            _backend: PhantomData,
        }
    }

    fn load_record(self, record: <Self::PolicyState as PolicyState<B>>::Record) -> Self {
        let state = self.state().load_record(record);
        Self {
            model: state.model,
            _backend: PhantomData,
        }
    }
}

#[derive(Record)]
pub struct DqnLearningRecord<B: AutodiffBackend, M: AutodiffModule<B>, O: Optimizer<M, B>> {
    policy_model: M::Record,
    target_model: M::Record,
    optimizer: O::Record,
}

#[derive(Clone)]
pub struct DqnLearningAgent<B, M, O>
where
    B: AutodiffBackend,
    M: DiscreteActionModel<B> + AutodiffModule<B> + TargetModel<B> + 'static,
    M::InnerModule: DiscreteActionModel<B::InnerBackend> + TargetModel<B::InnerBackend>,
    O: Optimizer<M, B> + 'static,
{
    policy_model: M,
    target_model: M,
    agent: EpsilonGreedyPolicy<B, DQN<B, M>>,
    optimizer: O,
    config: DqnAgentConfig,
}

impl<B, M, O> DqnLearningAgent<B, M, O>
where
    B: AutodiffBackend,
    M: DiscreteActionModel<B> + AutodiffModule<B> + TargetModel<B> + 'static,
    M::InnerModule: DiscreteActionModel<B::InnerBackend> + TargetModel<B::InnerBackend>,
    O: Optimizer<M, B> + 'static,
{
    pub fn new(model: M, optimizer: O, config: DqnAgentConfig) -> Self {
        let agent = EpsilonGreedyPolicy::new(
            DQN::new(model.clone()),
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay,
        );
        Self {
            policy_model: model.clone(),
            target_model: model,
            agent,
            optimizer: optimizer,
            config,
        }
    }
}

#[derive(Clone)]
pub struct SimpleTrainOutput<B: Backend> {
    pub policy_model_loss: Tensor<B, 1>,
}

impl<B: Backend> ItemLazy for SimpleTrainOutput<B> {
    type ItemSync = SimpleTrainOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [loss] = Transaction::default()
            .register(self.policy_model_loss)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        SimpleTrainOutput {
            policy_model_loss: Tensor::from_data(loss, device),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for SimpleTrainOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.policy_model_loss.clone())
    }
}

impl<B, M, O> PolicyLearner<B> for DqnLearningAgent<B, M, O>
where
    B: AutodiffBackend,
    M: DiscreteActionModel<B> + AutodiffModule<B> + TargetModel<B> + 'static,
    M::Input: Clone,
    M::InnerModule: DiscreteActionModel<B::InnerBackend> + TargetModel<B::InnerBackend>,
    O: Optimizer<M, B> + 'static,
{
    type TrainContext = SimpleTrainOutput<B>;
    type InnerPolicy = EpsilonGreedyPolicy<B, DQN<B, M>>;
    type Record = DqnLearningRecord<B, M, O>;

    // Not that complex. No use in extracting in a type alias that is going to be used only once.
    #[allow(clippy::type_complexity)]
    fn train(
        &mut self,
        input: TransitionBatch<
            B,
            <Self::InnerPolicy as Policy<B>>::Observation,
            <Self::InnerPolicy as Policy<B>>::Action,
        >,
    ) -> RLTrainOutput<Self::TrainContext, <Self::InnerPolicy as Policy<B>>::PolicyState> {
        let states_batch = input.states;
        let next_states_batch = input.next_states;
        let actions_batch = input.actions.actions;
        let rewards_batch = input.rewards;
        let dones_batch = input.dones;

        // Optimize
        let logits = self.policy_model.forward(states_batch).logits;
        let state_action_values = logits.gather(1, actions_batch.int());

        let next_state_values = self.target_model.forward(next_states_batch.clone());
        let next_state_values = next_state_values.logits.max_dim(1).squeeze::<1>();

        let not_done_batch = Tensor::ones_like(&dones_batch) - dones_batch;
        let expected_state_action_values = (next_state_values * not_done_batch.squeeze())
            .mul_scalar(self.config.gamma)
            + rewards_batch.squeeze();
        let expected_state_action_values = expected_state_action_values.unsqueeze_dim::<2>(1);

        let loss = MseLoss::new().forward(
            state_action_values,
            expected_state_action_values,
            nn::loss::Reduction::Mean,
        );
        let gradients = loss.backward();
        let gradient_params = GradientsParams::from_grads(gradients, &self.policy_model);
        self.policy_model = self.optimizer.step(
            self.config.learning_rate,
            self.policy_model.clone(),
            gradient_params,
        );
        self.target_model = self
            .target_model
            .soft_update(&self.policy_model, self.config.tau);
        let policy_update = EpsilonGreedyPolicyState::new(
            DqnState {
                model: self.policy_model.clone(),
                _backend: PhantomData,
            },
            self.agent.state().step,
        );
        self.agent.update(policy_update.clone());
        RLTrainOutput {
            policy: policy_update,
            item: SimpleTrainOutput {
                policy_model_loss: loss,
            },
        }
    }

    fn policy(&self) -> Self::InnerPolicy {
        self.agent.clone()
    }

    fn update_policy(&mut self, update: Self::InnerPolicy) {
        self.agent = update;
    }

    fn record(&self) -> Self::Record {
        DqnLearningRecord {
            policy_model: self.policy_model.clone().into_record(),
            target_model: self.target_model.clone().into_record(),
            optimizer: self.optimizer.to_record(),
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        let policy_model = self.policy_model.load_record(record.policy_model);
        let target_model = self.target_model.load_record(record.target_model);
        let optimizer = self.optimizer.load_record(record.optimizer);
        Self {
            policy_model,
            target_model,
            agent: self.agent,
            optimizer,
            config: self.config,
        }
    }
}

use burn::module::Module;
use burn::store::{ModuleRecord, RecordError};
use burn::rl::{
    Batchable, LearnerTransitionBatch, Policy, PolicyLearner, PolicyState, RLTrainOutput,
    SliceAccess,
};
use burn::tensor::activation::softmax;
use burn::tensor::{Bytes, Device, Int, Transaction};
use burn::train::ItemLazy;
use burn::train::checkpoint::{Checkpoint, CheckpointerError};
use burn::train::metric::{Adaptor, LossInput};
use burn::{
    Tensor,
    config::Config,
    module::AutodiffModule,
    nn::{self, loss::MseLoss},
    optim::{GradientsParams, ModuleOptimizer, OptimizerRecord},
};
use std::path::PathBuf;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rng;

use crate::utils::{
    EpsilonGreedyPolicy, EpsilonGreedyPolicyState, create_lin_layers, soft_update_linear,
};

pub trait DiscreteActionModel: Module {
    type Input: Clone + Send + Batchable;

    fn forward(&self, input: Self::Input) -> DiscreteLogitsTensor<2>;
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
pub struct MlpNet {
    pub linears: Vec<nn::Linear>,
    pub dropout: nn::Dropout,
    pub activation: nn::Relu,
}

impl MlpNet {
    /// Create the module from the given configuration.
    pub fn new(config: &MlpNetConfig, device: &Device) -> Self {
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
pub struct ObservationTensor<const D: usize> {
    pub state: Tensor<D>,
}

impl<const D: usize> Batchable for ObservationTensor<D> {
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

impl SliceAccess for ObservationTensor<2> {
    fn zeros_like(sample: &Self, capacity: usize, device: &Device) -> Self {
        let feature_dim = sample.state.dims()[1];
        Self {
            state: Tensor::zeros([capacity, feature_dim], device),
        }
    }

    fn select(self, dim: usize, indices: Tensor<1, Int>) -> Self {
        Self {
            state: Tensor::select(self.state, dim, indices),
        }
    }

    fn slice_assign_inplace(&mut self, index: usize, value: Self) {
        // The given value needs to be on the same backend as the state tensor.
        let device = self.state.device();
        let state = if value.state.device().is_autodiff() {
            value.state.inner()
        } else {
            value.state
        };
        let state = if device.is_autodiff() {
            Tensor::from_inner(state.to_device(&device.inner()))
        } else {
            state.to_device(&device.inner())
        };
        self.state
            .inplace(|t| t.slice_assign(index..index + 1, state));
    }
}

impl DiscreteActionModel for MlpNet {
    type Input = ObservationTensor<2>;

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, d_input]`
    /// - output: `[batch_size, d_output]`
    fn forward(&self, input: Self::Input) -> DiscreteLogitsTensor<2> {
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

pub trait TargetModel {
    fn soft_update(&self, that: &Self, tau: f64) -> Self;
}

impl TargetModel for MlpNet {
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
pub struct DqnState<M: DiscreteActionModel> {
    model: M,
}

impl<M: DiscreteActionModel> PolicyState for DqnState<M> {
    type Record = ModuleRecord;

    fn into_record(self) -> Self::Record {
        self.model.clone().into_record()
    }

    fn load_record(&self, record: Self::Record) -> Self {
        Self {
            model: self.model.clone().load_record(record),
        }
    }
}

#[derive(Clone)]
pub struct DQN<M: DiscreteActionModel> {
    model: M,
}

impl<M: DiscreteActionModel> DQN<M> {
    pub fn new(policy: M) -> Self {
        Self { model: policy }
    }

    pub fn model(&self) -> M {
        self.model.clone()
    }

    pub fn set_model(&mut self, model: M) {
        self.model = model;
    }
}

#[derive(Clone)]
pub struct DiscreteLogitsTensor<const D: usize> {
    pub logits: Tensor<D>,
}

impl<const D: usize> Batchable for DiscreteLogitsTensor<D> {
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
pub struct DiscreteActionTensor<const D: usize> {
    pub actions: Tensor<D>,
}

impl<const D: usize> Batchable for DiscreteActionTensor<D> {
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

impl SliceAccess for DiscreteActionTensor<2> {
    fn zeros_like(sample: &Self, capacity: usize, device: &Device) -> Self {
        let feature_dim = sample.actions.dims()[1];
        Self {
            actions: Tensor::zeros([capacity, feature_dim], device),
        }
    }

    fn select(self, dim: usize, indices: Tensor<1, Int>) -> Self {
        Self {
            actions: Tensor::select(self.actions, dim, indices),
        }
    }

    fn slice_assign_inplace(&mut self, index: usize, value: Self) {
        // The given value needs to be on the same backend as the state tensor.
        let device = self.actions.device();
        let actions = if value.actions.device().is_autodiff() {
            value.actions.inner()
        } else {
            value.actions
        };
        let actions = if device.is_autodiff() {
            Tensor::from_inner(actions.to_device(&device.inner()))
        } else {
            actions.to_device(&device.inner())
        };
        self.actions
            .inplace(|t| t.slice_assign(index..index + 1, actions));
    }
}

impl<M: DiscreteActionModel> Policy for DQN<M> {
    type Observation = M::Input;
    type ActionDistribution = DiscreteLogitsTensor<2>;
    type Action = DiscreteActionTensor<2>;

    type ActionContext = ();
    type PolicyState = DqnState<M>;

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
        for p in probs {
            let dist = WeightedIndex::new(p.to_data().to_vec::<f32>().unwrap()).unwrap();
            let action = dist.sample(&mut rng);
            actions.push(Tensor::<1>::from_floats([action], &p.device()));
        }

        let output = DiscreteActionTensor {
            actions: Tensor::stack(actions, 1),
        };
        (output, vec![])
    }

    fn update(&mut self, update: Self::PolicyState) {
        self.model = update.model;
    }

    fn state(&self) -> Self::PolicyState {
        DqnState {
            model: self.model.clone(),
        }
    }

    fn to_device(self, device: &Device) -> Self {
        Self {
            model: self.model.to_device(device),
        }
    }

    fn load_record(self, record: <Self::PolicyState as PolicyState>::Record) -> Self {
        let state = self.state().load_record(record);
        Self { model: state.model }
    }
}

/// The learner state, persisted as a single burnpack checkpoint file.
///
/// It bundles the policy and target [`ModuleRecord`]s together with the optimizer
/// [`OptimizerRecord`]. All three are device-free; the optimizer tensors are re-materialized on a
/// device when the agent loads the record (see [`DqnLearningAgent::load_record`]).
pub struct DqnLearningRecord {
    policy_model: ModuleRecord,
    target_model: ModuleRecord,
    optimizer: OptimizerRecord,
}

/// Encode a little-endian `u64` length prefix followed by the bytes themselves.
fn frame(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Read a length-prefixed frame, returning the slice and advancing `offset`.
fn unframe<'a>(data: &'a [u8], offset: &mut usize) -> Result<&'a [u8], CheckpointerError> {
    let header_end = *offset + 8;
    if data.len() < header_end {
        return Err(CheckpointerError::Unknown(
            "Corrupted checkpoint: missing length prefix.".to_string(),
        ));
    }
    let len = u64::from_le_bytes(data[*offset..header_end].try_into().unwrap()) as usize;
    let body_end = header_end + len;
    if data.len() < body_end {
        return Err(CheckpointerError::Unknown(
            "Corrupted checkpoint: truncated frame.".to_string(),
        ));
    }
    *offset = body_end;
    Ok(&data[header_end..body_end])
}

fn record_err(err: RecordError) -> CheckpointerError {
    CheckpointerError::Record(err)
}

impl Checkpoint for DqnLearningRecord {
    fn save(self, path: PathBuf) -> Result<(), CheckpointerError> {
        let policy = self.policy_model.into_bytes().map_err(record_err)?;
        let target = self.target_model.into_bytes().map_err(record_err)?;
        let optimizer = self.optimizer.into_bytes().map_err(record_err)?;

        let mut out = Vec::new();
        frame(&mut out, &policy);
        frame(&mut out, &target);
        frame(&mut out, &optimizer);

        std::fs::write(path, out).map_err(CheckpointerError::IOError)
    }

    fn load(path: PathBuf) -> Result<Self, CheckpointerError> {
        let data = std::fs::read(path).map_err(CheckpointerError::IOError)?;
        let mut offset = 0;

        let policy = unframe(&data, &mut offset)?;
        let policy_model =
            ModuleRecord::from_bytes(Bytes::from_bytes_vec(policy.to_vec())).map_err(record_err)?;

        let target = unframe(&data, &mut offset)?;
        let target_model =
            ModuleRecord::from_bytes(Bytes::from_bytes_vec(target.to_vec())).map_err(record_err)?;

        let optimizer = unframe(&data, &mut offset)?;
        let optimizer =
            OptimizerRecord::from_bytes(Bytes::from_bytes_vec(optimizer.to_vec())).map_err(record_err)?;

        Ok(Self {
            policy_model,
            target_model,
            optimizer,
        })
    }
}

#[derive(Clone)]
pub struct DqnLearningAgent<M>
where
    M: DiscreteActionModel + AutodiffModule + TargetModel + 'static,
{
    policy_model: M,
    target_model: M,
    agent: EpsilonGreedyPolicy<DQN<M>>,
    optimizer: ModuleOptimizer,
    config: DqnAgentConfig,
}

impl<M> DqnLearningAgent<M>
where
    M: DiscreteActionModel + AutodiffModule + TargetModel + 'static,
{
    pub fn new(model: M, optimizer: ModuleOptimizer, config: DqnAgentConfig) -> Self {
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
            optimizer,
            config,
        }
    }
}

#[derive(Clone)]
pub struct SimpleTrainOutput {
    pub policy_model_loss: Tensor<1>,
}

impl ItemLazy for SimpleTrainOutput {
    fn sync(self) -> Self {
        let [loss] = Transaction::default()
            .register(self.policy_model_loss)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Device::flex();

        SimpleTrainOutput {
            policy_model_loss: Tensor::from_data(loss, device),
        }
    }
}

impl Adaptor<LossInput> for SimpleTrainOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.policy_model_loss.clone())
    }
}

impl<M> PolicyLearner for DqnLearningAgent<M>
where
    M: DiscreteActionModel + AutodiffModule + TargetModel + 'static,
    M::Input: Clone,
{
    type TrainContext = SimpleTrainOutput;
    type InnerPolicy = EpsilonGreedyPolicy<DQN<M>>;
    type Record = DqnLearningRecord;

    fn train(
        &mut self,
        input: LearnerTransitionBatch<Self::InnerPolicy>,
    ) -> RLTrainOutput<Self::TrainContext, <Self::InnerPolicy as Policy>::PolicyState> {
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
        let mut policy = self.agent.clone();
        let mut inner_policy = policy.inner_policy();
        inner_policy.set_model(policy.inner_policy().model().valid());
        policy.set_inner_policy(inner_policy);
        policy
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

    fn device(&self) -> Device {
        self.policy_model.devices()[0].clone()
    }
}

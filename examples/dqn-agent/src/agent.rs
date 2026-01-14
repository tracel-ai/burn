use burn::backend::NdArray;
use burn::module::Module;
use burn::optim::SimpleOptimizer;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::tensor::{Device, Transaction, s};
use burn::train::ItemLazy;
use burn::train::metric::{Adaptor, ExplorationRateInput, LossInput};
use burn::{
    Tensor,
    config::Config,
    module::AutodiffModule,
    nn::{self, loss::MseLoss},
    optim::{GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{Distribution, backend::AutodiffBackend},
};
use burn_rl::{
    ActionContext, Agent, EnvAction, EnvState, Environment, LearnerAgent, RLTrainOutput,
    Transition, TransitionBatch, TransitionBuffer,
};
use std::marker::PhantomData;

use crate::utils::{EpsilonGreedyPolicy, create_lin_layers, soft_update_linear};

#[derive(Clone)]
pub struct InferenceOutput<B: Backend> {
    pub probs: Tensor<B, 2>,
    pub values: Option<Tensor<B, 2>>,
}

// TODO : What about infer retourne aussi TO. batch_take_action pourrait aussi retourner TO et le mettre dans la TransitionBatch or smtg.
pub trait AgentModel<B: Backend>: Module<B> {
    fn forward(&self, input: Tensor<B, 2>) -> InferenceOutput<B>;
    fn infer(&self, input: Tensor<B, 2>) -> InferenceOutput<B>;
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

impl<B: Backend> AgentModel<B> for MlpNet<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, d_input]`
    /// - output: `[batch_size, d_output]`
    fn forward(&self, input: Tensor<B, 2>) -> InferenceOutput<B> {
        let mut x = input;

        for (i, linear) in self.linears.iter().enumerate() {
            x = linear.forward(x);
            x = self.dropout.forward(x);
            if i < self.linears.len() - 1 {
                x = self.activation.forward(x);
            }
        }

        InferenceOutput {
            probs: x,
            values: None,
        }
    }

    fn infer(&self, input: Tensor<B, 2>) -> InferenceOutput<B> {
        self.forward(input)
    }

    // fn infer(&self, input: Tensor<B, 2>) -> InferenceOutput<B, Self::ForwardOutput> {
    //     let x = self.forward(input);
    //     InferenceOutput {
    //         logits: x.clone(),
    //         training_output: x,
    //     }
    // }
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

pub trait DqnModel<B: Backend> {
    fn soft_update(&self, that: &Self, tau: f64) -> Self;
}

impl<B: Backend> DqnModel<B> for MlpNet<B> {
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
pub struct DqnAgent<B: Backend, E: Environment, M: AgentModel<B> + DqnModel<B>> {
    policy: M,
    exploration_policy: EpsilonGreedyPolicy,
    device: Device<B>,
    _env: PhantomData<E::State>,
}

impl<B: Backend, E: Environment, M: AgentModel<B> + DqnModel<B>> DqnAgent<B, E, M> {
    pub fn new(policy: M, exploration_policy: EpsilonGreedyPolicy, device: &Device<B>) -> Self {
        Self {
            policy,
            exploration_policy,
            device: device.clone(),
            _env: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct EpsilonGreedyPolicyOutput {
    pub epsilon: f64,
}

impl ItemLazy for EpsilonGreedyPolicyOutput {
    type ItemSync = EpsilonGreedyPolicyOutput;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl Adaptor<ExplorationRateInput> for EpsilonGreedyPolicyOutput {
    fn adapt(&self) -> ExplorationRateInput {
        ExplorationRateInput::new(self.epsilon)
    }
}

// TODO: remove Environment
impl<B: Backend, E: Environment, M: AgentModel<B> + DqnModel<B> + Clone + Send> Agent<B, E>
    for DqnAgent<B, E, M>
{
    type Policy = M;
    type DecisionContext = EpsilonGreedyPolicyOutput;

    fn batch_take_action(
        &mut self,
        states: Vec<E::State>,
        deterministic: bool,
    ) -> Vec<ActionContext<E::Action, EpsilonGreedyPolicyOutput>> {
        let input = states.iter().map(|s| s.to_tensor(&self.device)).collect();
        let input = Tensor::stack(input, 0);
        let InferenceOutput {
            probs: logits,
            values: _,
        } = self.policy.forward(input);
        let actions = logits.argmax(1);

        if deterministic {
            let actions = (0..actions.dims()[0])
                .map(|i| {
                    ActionContext::new(
                        EpsilonGreedyPolicyOutput { epsilon: 0.0 },
                        E::Action::from_tensor(actions.clone().float().slice(s![i, ..])),
                    )
                })
                .collect();
            return actions;
        }

        // Epsilon greedy policy
        let bs = actions.shape()[0];
        let distribution = Distribution::Bernoulli(0.5);
        let device = self.device.clone();
        let rand_actions = Tensor::random([bs, 1], distribution, &device);

        let threshold = self.exploration_policy.step();
        let distribution = Distribution::Bernoulli(threshold);
        let random_indices = Tensor::random([bs, 1], distribution, &device);
        let inv_mask = random_indices.clone().neg().add_scalar(1);
        let actions = (actions.clone() * inv_mask.clone() + rand_actions * random_indices).float();
        (0..actions.dims()[0])
            .map(|i| {
                ActionContext::new(
                    EpsilonGreedyPolicyOutput { epsilon: threshold },
                    E::Action::from_tensor(actions.clone().slice(s![i, ..])),
                )
            })
            .collect()
    }

    fn update_policy(&mut self, update: Self::Policy) {
        self.policy = update;
    }

    fn take_action(
        &mut self,
        state: E::State,
        deterministic: bool,
    ) -> ActionContext<E::Action, Self::DecisionContext> {
        self.batch_take_action(vec![state], deterministic)[0].clone()
    }
}

#[derive(Clone)]
pub struct DqnLearningAgent<B, E, M, O>
where
    B: AutodiffBackend,
    E: Environment,
    M: AgentModel<B> + AutodiffModule<B> + DqnModel<B> + 'static,
    M::InnerModule: AgentModel<B::InnerBackend> + DqnModel<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend> + 'static,
{
    policy_model: M,
    target_model: M,
    agent: DqnAgent<B::InnerBackend, E, M::InnerModule>,
    optimizer: OptimizerAdaptor<O, M, B>,
    config: DqnAgentConfig,
}

impl<B, E, M, O> DqnLearningAgent<B, E, M, O>
where
    B: AutodiffBackend,
    E: Environment,
    M: AgentModel<B> + AutodiffModule<B> + DqnModel<B> + 'static,
    M::InnerModule: AgentModel<B::InnerBackend> + DqnModel<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend> + 'static,
{
    pub fn new(
        model: M,
        optimizer: OptimizerAdaptor<O, M, B>,
        config: DqnAgentConfig,
        exploration_policy: EpsilonGreedyPolicy,
        device: &Device<B::InnerBackend>,
    ) -> Self {
        let agent = DqnAgent::new(model.valid(), exploration_policy, device);
        Self {
            policy_model: model.clone(),
            target_model: model,
            agent,
            optimizer: optimizer.into(),
            config,
        }
    }
}

impl<B, E, M, O> Agent<B, E> for DqnLearningAgent<B, E, M, O>
where
    B: AutodiffBackend,
    E: Environment,
    M: AgentModel<B> + AutodiffModule<B> + DqnModel<B> + 'static,
    M::InnerModule: AgentModel<B::InnerBackend> + DqnModel<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend> + 'static,
{
    type Policy = M;
    type DecisionContext = EpsilonGreedyPolicyOutput;

    fn batch_take_action(
        &mut self,
        states: Vec<E::State>,
        deterministic: bool,
    ) -> Vec<ActionContext<E::Action, EpsilonGreedyPolicyOutput>> {
        self.agent.batch_take_action(states, deterministic)
    }

    fn update_policy(&mut self, update: Self::Policy) {
        self.agent.update_policy(update.valid());
    }

    fn take_action(
        &mut self,
        state: <E as Environment>::State,
        deterministic: bool,
    ) -> ActionContext<<E as Environment>::Action, Self::DecisionContext> {
        self.batch_take_action(vec![state], deterministic)[0].clone()
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

impl<B, E, M, O> LearnerAgent<B, E> for DqnLearningAgent<B, E, M, O>
where
    B: AutodiffBackend,
    E: Environment,
    M: AgentModel<B> + AutodiffModule<B> + DqnModel<B> + 'static,
    M::InnerModule: AgentModel<B::InnerBackend> + DqnModel<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend> + 'static,
{
    type TrainingInput = Transition<B>;
    type TrainingOutput = SimpleTrainOutput<B>;

    fn train(
        &mut self,
        input: &TransitionBuffer<Self::TrainingInput>,
    ) -> RLTrainOutput<Self::TrainingOutput, Self::Policy> {
        // TODO: true batch size.
        let batch = input.random_sample(128);
        let batch = TransitionBatch::from(batch);

        let states_batch = batch.states;
        let next_states_batch = batch.next_states;
        let actions_batch = batch.actions;
        let rewards_batch = batch.rewards;
        let dones_batch = batch.dones;

        // Optimize
        let InferenceOutput {
            probs: logits,
            values: _,
        } = self.policy_model.forward(states_batch.clone());
        let state_action_values = logits.gather(1, actions_batch.int());

        let InferenceOutput {
            probs: next_state_values,
            values: _,
        } = self.target_model.forward(next_states_batch.clone());
        let next_state_values = next_state_values.max_dim(1).squeeze::<1>();

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
        RLTrainOutput {
            policy: self.policy(),
            item: SimpleTrainOutput {
                policy_model_loss: loss,
            },
        }
    }

    fn policy(&self) -> Self::Policy {
        self.policy_model.clone()
    }
}

use burn::backend::NdArray;
use burn::module::Module;
use burn::optim::SimpleOptimizer;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::tensor::activation::softmax;
use burn::tensor::{Device, Transaction, s};
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
use burn_rl::{
    ActionContext, EnvAction, EnvState, Environment, LearnerAgent, Policy, RLTrainOutput,
    Transition, TransitionBatch, TransitionBuffer,
};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rng;
use std::marker::PhantomData;

use crate::utils::{
    EpsilonGreedyPolicy, EpsilonGreedyPolicyState, create_lin_layers, soft_update_linear,
};

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
    model: M,
    device: Device<B>,
    _env: PhantomData<E::State>,
}

impl<B: Backend, E: Environment, M: AgentModel<B> + DqnModel<B>> DqnAgent<B, E, M> {
    pub fn new(policy: M, device: &Device<B>) -> Self {
        Self {
            model: policy,
            device: device.clone(),
            _env: PhantomData,
        }
    }
}

// TODO: remove Environment
impl<B: Backend, E: Environment, M: AgentModel<B> + DqnModel<B> + Clone + Send>
    Policy<B, E::State, E::Action> for DqnAgent<B, E, M>
{
    type ActionContext = ();
    type PolicyState = M;

    fn logits(&mut self, state: &E::State) -> Tensor<B, 1> {
        self.batch_logits(vec![state]).squeeze()
    }

    fn action(
        &mut self,
        state: &E::State,
        deterministic: bool,
    ) -> ActionContext<E::Action, Self::ActionContext> {
        self.batch_action(vec![state], deterministic).remove(0)
    }

    fn batch_logits(&mut self, states: Vec<&E::State>) -> Tensor<B, 2> {
        let input = states.iter().map(|s| s.to_tensor(&self.device)).collect();
        let input = Tensor::stack(input, 0);
        let InferenceOutput {
            probs: logits,
            values: _,
        } = self.model.forward(input);
        logits
    }

    fn batch_action(
        &mut self,
        states: Vec<&E::State>,
        deterministc: bool,
    ) -> Vec<ActionContext<E::Action, Self::ActionContext>> {
        let logits = self.batch_logits(states);
        if deterministc {
            let actions = logits.argmax(1);
            let actions = (0..actions.dims()[0])
                .map(|i| {
                    ActionContext::new(
                        (),
                        E::Action::from_tensor(actions.clone().float().slice(s![i, ..])),
                    )
                })
                .collect();
            return actions;
        }

        let mut actions = vec![];
        let probs = softmax(logits, 1);
        let mut rng = rng();
        for i in 0..probs.dims()[0] {
            let dist = WeightedIndex::new(
                probs
                    .clone()
                    .slice(s![i, ..])
                    .squeeze::<1>()
                    .to_data()
                    .to_vec::<f32>()
                    .unwrap(),
            )
            .unwrap();
            let action = dist.sample(&mut rng);
            actions.push(ActionContext::new((), E::Action::from_usize(action)));
        }
        return actions;
    }

    fn update(&mut self, update: Self::PolicyState) {
        self.model = update;
    }

    fn state(&self) -> Self::PolicyState {
        self.model.clone()
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
    agent: EpsilonGreedyPolicy<B, E::State, E::Action, DqnAgent<B, E, M>>,
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
        device: &Device<B>,
    ) -> Self {
        let agent = EpsilonGreedyPolicy::new(
            DqnAgent::new(model.clone(), device),
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay,
        );
        Self {
            policy_model: model.clone(),
            target_model: model,
            agent,
            optimizer: optimizer.into(),
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

impl<B, E, M, O> LearnerAgent<B, E::State, E::Action> for DqnLearningAgent<B, E, M, O>
where
    B: AutodiffBackend,
    E: Environment,
    M: AgentModel<B> + AutodiffModule<B> + DqnModel<B> + 'static,
    M::InnerModule: AgentModel<B::InnerBackend> + DqnModel<B::InnerBackend>,
    O: SimpleOptimizer<B::InnerBackend> + 'static,
{
    type TrainingInput = Transition<B>;
    type TrainingOutput = SimpleTrainOutput<B>;
    type InnerPolicy = EpsilonGreedyPolicy<B, E::State, E::Action, DqnAgent<B, E, M>>;

    fn train(
        &mut self,
        input: &TransitionBuffer<Self::TrainingInput>,
    ) -> RLTrainOutput<
        Self::TrainingOutput,
        <Self::InnerPolicy as Policy<B, E::State, E::Action>>::PolicyState,
    > {
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
        let policy_update =
            EpsilonGreedyPolicyState::new(self.policy_model.clone(), self.agent.state().step);
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
}

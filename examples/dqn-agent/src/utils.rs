use std::marker::PhantomData;

use burn::{
    Tensor,
    module::{Param, ParamId},
    nn::{self, Linear},
    prelude::Backend,
    record::Record,
    rl::{Policy, PolicyState},
    tensor::Device,
    train::{
        ItemLazy,
        metric::{Adaptor, ExplorationRateInput},
    },
};
use derive_new::new;
use rand::{random, random_range};

use crate::agent::{TensorActionOutput, TensorLogits};

pub fn create_lin_layers<B: Backend>(
    num_layers: usize,
    d_input: usize,
    d_hidden: usize,
    d_output: usize,
    device: &Device<B>,
) -> Vec<Linear<B>> {
    let mut linears = Vec::with_capacity(num_layers);

    if num_layers == 1 {
        linears.push(nn::LinearConfig::new(d_input, d_output).init(device));
        return linears;
    }
    for i in 0..num_layers {
        if i == 0 {
            linears.push(nn::LinearConfig::new(d_input, d_hidden).init(device));
        } else if i == num_layers - 1 {
            linears.push(nn::LinearConfig::new(d_hidden, d_output).init(device));
        } else {
            linears.push(nn::LinearConfig::new(d_hidden, d_hidden).init(device));
        }
    }
    linears
}

pub fn soft_update_linear<B: Backend>(this: Linear<B>, that: &Linear<B>, tau: f64) -> Linear<B> {
    let weight = soft_update_tensor(&this.weight, &that.weight, tau);
    let bias = match (&this.bias, &that.bias) {
        (Some(this_bias), Some(that_bias)) => Some(soft_update_tensor(this_bias, that_bias, tau)),
        _ => None,
    };

    Linear::<B> { weight, bias }
}

fn soft_update_tensor<const N: usize, B: Backend>(
    this: &Param<Tensor<B, N>>,
    that: &Param<Tensor<B, N>>,
    tau: f64,
) -> Param<Tensor<B, N>> {
    let that_weight = that.val();
    let this_weight = this.val();
    let new_weight = this_weight * (1.0 - tau) + that_weight * tau;

    Param::initialized(ParamId::new(), new_weight)
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

#[derive(Record)]
pub struct EpsilonGreedyPolicyRecord<B: Backend, P: Policy<B>> {
    pub inner_state: <P::PolicyState as PolicyState<B>>::Record,
    pub step: usize,
}

#[derive(Clone, new)]
pub struct EpsilonGreedyPolicyState<B: Backend, P: Policy<B>> {
    pub inner_state: P::PolicyState,
    pub step: usize,
}

impl<B: Backend, P: Policy<B>> PolicyState<B> for EpsilonGreedyPolicyState<B, P> {
    type Record = EpsilonGreedyPolicyRecord<B, P>;

    fn into_record(&self) -> Self::Record {
        EpsilonGreedyPolicyRecord {
            inner_state: self.inner_state.into_record(),
            step: self.step,
        }
    }

    fn load_record(&self, record: Self::Record) -> Self {
        let inner_state = self.inner_state.load_record(record.inner_state);
        Self {
            inner_state,
            step: record.step,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EpsilonGreedyPolicy<B: Backend, P: Policy<B>> {
    inner_policy: P,
    eps_start: f64,
    eps_end: f64,
    eps_decay: f64,
    step: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend, P: Policy<B>> EpsilonGreedyPolicy<B, P> {
    pub fn new(inner_policy: P, eps_start: f64, eps_end: f64, eps_decay: f64) -> Self {
        Self {
            inner_policy,
            eps_start,
            eps_end,
            eps_decay,
            step: 0,
            _backend: PhantomData,
        }
    }

    fn get_threshold(&self) -> f64 {
        self.eps_end
            + (self.eps_start - self.eps_end) * f64::exp(-1. * self.step as f64 / self.eps_decay)
    }

    fn step(&mut self) -> f64 {
        let thresh = self.get_threshold();
        self.step += 1;
        thresh
    }
}

impl<B, P> Policy<B> for EpsilonGreedyPolicy<B, P>
where
    B: Backend,
    P: Policy<B, Output = TensorLogits<B, 2>, Action = TensorActionOutput<B, 2>>,
{
    type ActionContext = EpsilonGreedyPolicyOutput;
    type PolicyState = EpsilonGreedyPolicyState<B, P>;

    type Input = P::Input;
    type Output = TensorLogits<B, 2>;
    type Action = TensorActionOutput<B, 2>;

    fn forward(&mut self, states: Self::Input) -> Self::Output {
        self.inner_policy.forward(states)
    }

    fn action(
        &mut self,
        states: Self::Input,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>) {
        let logits = self.inner_policy.forward(states).logits;
        let greedy_actions = logits.argmax(1);
        let greedy_actions = greedy_actions.split(1, 0);

        let mut contexts = vec![];
        let mut actions = vec![];
        for i in 0..greedy_actions.len() {
            let threshold = self.step();
            let threshold = if deterministic { 0.0 } else { threshold };
            contexts.push(EpsilonGreedyPolicyOutput { epsilon: threshold });
            if random::<f64>() > threshold {
                actions.push(greedy_actions[i].clone().float());
            } else {
                actions.push(
                    Tensor::<B, 1>::from_floats([random_range(0..2)], &greedy_actions[i].device())
                        .unsqueeze(),
                );
            }
        }

        let output = Tensor::cat(actions, 0);
        (TensorActionOutput { actions: output }, contexts)
    }

    fn update(&mut self, update: Self::PolicyState) {
        // Note : updating an epsilon greedy policy doesn't change the step.
        self.inner_policy.update(update.inner_state);
    }

    fn state(&self) -> Self::PolicyState {
        EpsilonGreedyPolicyState {
            inner_state: self.inner_policy.state(),
            step: self.step,
        }
    }

    fn from_record(&self, record: <Self::PolicyState as PolicyState<B>>::Record) -> Self {
        let state = self.state().load_record(record);
        let inner_policy = self
            .inner_policy
            .from_record(state.inner_state.into_record());
        EpsilonGreedyPolicy {
            inner_policy: inner_policy,
            eps_start: self.eps_start,
            eps_end: self.eps_end,
            eps_decay: self.eps_decay,
            step: state.step,
            _backend: PhantomData,
        }
    }
}

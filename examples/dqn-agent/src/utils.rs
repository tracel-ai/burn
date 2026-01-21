use std::marker::PhantomData;

use burn::{
    Tensor,
    module::{Param, ParamId},
    nn::{self, Linear},
    prelude::Backend,
    tensor::{Device, s},
    train::{
        ItemLazy,
        metric::{Adaptor, ExplorationRateInput},
    },
};
use burn_rl::Policy;
use derive_new::new;
use rand::{random, random_range};

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

#[derive(Clone, new)]
pub struct EpsilonGreedyPolicyState<B: Backend, P: Policy<B>> {
    pub inner_policy: P::PolicyState,
    pub step: usize,
    _backend: PhantomData<B>,
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
    P: Policy<B, Logits = Tensor<B, 2>, Action = Tensor<B, 2>>,
{
    type ActionContext = EpsilonGreedyPolicyOutput;
    type PolicyState = EpsilonGreedyPolicyState<B, P>;

    type Input = P::Input;
    type Logits = Tensor<B, 2>;
    type Action = Tensor<B, 2>;

    fn logits(&mut self, states: Self::Input) -> Self::Logits {
        self.inner_policy.logits(states)
    }

    fn action(
        &mut self,
        states: Self::Input,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>) {
        let logits = self.inner_policy.logits(states);
        let greedy_actions = logits.argmax(1);

        let mut contexts = vec![];
        let mut actions = vec![];
        for i in 0..greedy_actions.dims()[0] {
            let threshold = self.step();
            let threshold = if deterministic { 0.0 } else { threshold };
            contexts.push(EpsilonGreedyPolicyOutput { epsilon: threshold });
            if random::<f64>() > threshold {
                actions.push(greedy_actions.clone().slice(s![i, ..]).float());
            } else {
                actions.push(
                    Tensor::<B, 1>::from_floats([random_range(0..2)], &greedy_actions.device())
                        .unsqueeze(),
                );
            }
        }
        (Tensor::cat(actions, 0), contexts)
    }

    fn update(&mut self, update: Self::PolicyState) {
        // TODO: what to do
        // self.step = update.step;
        self.inner_policy.update(update.inner_policy);
    }

    fn state(&self) -> Self::PolicyState {
        EpsilonGreedyPolicyState {
            inner_policy: self.inner_policy.state(),
            step: self.step,
            _backend: PhantomData,
        }
    }

    fn batch(&self, inputs: Vec<&Self::Input>) -> Self::Input {
        self.inner_policy.batch(inputs)
    }

    fn unbatch(&self, inputs: Self::Action) -> Vec<Self::Action> {
        self.inner_policy.unbatch(inputs)
    }
}

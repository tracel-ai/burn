use burn::{
    Tensor,
    module::{Param, ParamId},
    nn::{self, Linear},
    rl::{Policy, PolicyState},
    tensor::Device,
    train::{
        ItemLazy,
        metric::{Adaptor, ExplorationRateInput},
    },
};
use derive_new::new;
use rand::{random, random_range};

use crate::agent::{DiscreteActionTensor, DiscreteLogitsTensor};

pub fn create_lin_layers(
    num_layers: usize,
    d_input: usize,
    d_hidden: usize,
    d_output: usize,
    device: &Device,
) -> Vec<Linear> {
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

pub fn soft_update_linear(this: Linear, that: &Linear, tau: f64) -> Linear {
    let weight = soft_update_tensor(&this.weight, &that.weight, tau);
    let bias = match (&this.bias, &that.bias) {
        (Some(this_bias), Some(that_bias)) => Some(soft_update_tensor(this_bias, that_bias, tau)),
        _ => None,
    };

    Linear { weight, bias }
}

fn soft_update_tensor<const N: usize>(
    this: &Param<Tensor<N>>,
    that: &Param<Tensor<N>>,
    tau: f64,
) -> Param<Tensor<N>> {
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
    fn sync(self) -> Self {
        self
    }
}

impl Adaptor<ExplorationRateInput> for EpsilonGreedyPolicyOutput {
    fn adapt(&self) -> ExplorationRateInput {
        ExplorationRateInput::new(self.epsilon)
    }
}

#[derive(Clone, new)]
pub struct EpsilonGreedyPolicyState<P: Policy> {
    pub inner_state: P::PolicyState,
    pub step: usize,
}

impl<P: Policy> PolicyState for EpsilonGreedyPolicyState<P> {
    // Only the inner policy's parameters are persisted; the exploration `step` counter resets on
    // load (it is part of the exploration schedule, not the learned parameters).
    type Record = <P::PolicyState as PolicyState>::Record;

    fn into_record(self) -> Self::Record {
        self.inner_state.into_record()
    }

    fn load_record(&self, record: Self::Record) -> Self {
        Self {
            inner_state: self.inner_state.load_record(record),
            step: self.step,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EpsilonGreedyPolicy<P: Policy> {
    inner_policy: P,
    eps_start: f64,
    eps_end: f64,
    eps_decay: f64,
    step: usize,
}

impl<P: Policy> EpsilonGreedyPolicy<P> {
    pub fn new(inner_policy: P, eps_start: f64, eps_end: f64, eps_decay: f64) -> Self {
        Self {
            inner_policy,
            eps_start,
            eps_end,
            eps_decay,
            step: 0,
        }
    }

    pub fn inner_policy(&self) -> P {
        self.inner_policy.clone()
    }

    pub fn set_inner_policy(&mut self, policy: P) {
        self.inner_policy = policy;
    }

    fn get_threshold(&self) -> f64 {
        self.eps_end
            + (self.eps_start - self.eps_end) * f64::exp(-(self.step as f64) / self.eps_decay)
    }

    fn step(&mut self) -> f64 {
        let thresh = self.get_threshold();
        self.step += 1;
        thresh
    }
}

impl<P> Policy for EpsilonGreedyPolicy<P>
where
    P: Policy<ActionDistribution = DiscreteLogitsTensor<2>, Action = DiscreteActionTensor<2>>,
{
    type ActionContext = EpsilonGreedyPolicyOutput;
    type PolicyState = EpsilonGreedyPolicyState<P>;

    type Observation = P::Observation;
    type ActionDistribution = DiscreteLogitsTensor<2>;
    type Action = DiscreteActionTensor<2>;

    fn forward(&mut self, states: Self::Observation) -> Self::ActionDistribution {
        self.inner_policy.forward(states)
    }

    fn action(
        &mut self,
        states: Self::Observation,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>) {
        let logits = self.inner_policy.forward(states).logits;
        let greedy_actions = logits.argmax(1);
        let greedy_actions = greedy_actions.split(1, 0);

        let mut contexts = vec![];
        let mut actions = vec![];
        for a in greedy_actions {
            let threshold = self.step();
            let threshold = if deterministic { 0.0 } else { threshold };
            contexts.push(EpsilonGreedyPolicyOutput { epsilon: threshold });
            if random::<f64>() > threshold {
                actions.push(a.clone().float().inner());
            } else {
                actions
                    .push(Tensor::<1>::from_floats([random_range(0..2)], &a.device()).unsqueeze());
            }
        }

        let output = Tensor::cat(actions, 0);
        (DiscreteActionTensor { actions: output }, contexts)
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

    fn to_device(self, device: &Device) -> Self {
        let mut policy = self.clone();
        let inner = policy.inner_policy().to_device(device);
        policy.set_inner_policy(inner);
        policy
    }

    fn load_record(self, record: <Self::PolicyState as PolicyState>::Record) -> Self {
        let state = self.state().load_record(record);
        let inner_policy = self
            .inner_policy
            .load_record(state.inner_state.into_record());
        EpsilonGreedyPolicy {
            inner_policy,
            eps_start: self.eps_start,
            eps_end: self.eps_end,
            eps_decay: self.eps_decay,
            step: state.step,
        }
    }
}

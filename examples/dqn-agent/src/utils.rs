use burn::{
    Tensor,
    module::{Param, ParamId},
    nn::{self, Linear},
    prelude::Backend,
    tensor::Device,
};

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

#[derive(Clone, Debug)]
pub struct EpsilonGreedyPolicy {
    eps_start: f64,
    eps_end: f64,
    eps_decay: f64,
    step: usize,
}

impl EpsilonGreedyPolicy {
    pub fn new(eps_start: f64, eps_end: f64, eps_decay: f64) -> Self {
        Self {
            eps_start,
            eps_end,
            eps_decay,
            step: 0,
        }
    }

    pub fn get_threshold(&self) -> f64 {
        self.eps_end
            + (self.eps_start - self.eps_end) * f64::exp(-1. * self.step as f64 / self.eps_decay)
    }

    pub fn step(&mut self) -> f64 {
        let thresh = self.get_threshold();
        self.step += 1;
        thresh
    }
}

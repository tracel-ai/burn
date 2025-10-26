use burn_core as burn;

use burn::{module::AutodiffModule, record::Record};

use burn::config::Config;
use burn::tensor::{Tensor, backend::AutodiffBackend};
use burn::tensor::{backend::Backend, ops::Device};

use super::{
    decay::{WeightDecay, WeightDecayConfig},
    momentum::{Momentum, MomentumConfig, MomentumState},
    SimpleOptimizer,
    adaptor::OptimizerAdaptor,
};
use crate::LearningRate;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Muon configuration.
/// 
/// Muon is an optimizer specifically designed for 2D parameters of neural network 
/// hidden layers (weight matrices). Other parameters such as biases and embeddings 
/// should be optimized using a standard method such as AdamW.
#[derive(Config, Debug)]
pub struct MuonConfig {
    /// [Weight decay](WeightDecayConfig) config.
    weight_decay: Option<WeightDecayConfig>,
    
    /// [Momentum](MomentumConfig) config.
    /// 
    /// Muon always uses momentum. Default configuration:
    /// - momentum: 0.95
    /// - dampening: 0.0
    /// - nesterov: true
    #[config(
        default = "MomentumConfig { momentum: 0.95, dampening: 0.0, nesterov: true }"
    )]
    momentum: MomentumConfig,
    
    /// Newton-Schulz iteration coefficients (a, b, c).
    #[config(default = "(3.4445, -4.775, 2.0315)")]
    ns_coefficients: (f32, f32, f32),

    /// Epsilon for numerical stability.
    #[config(default = 1e-7)]
    epsilon: f32,
    
    /// Number of Newton-Schulz iteration steps.
    #[config(default = 5)]
    ns_steps: usize,
}

/// Parameters for Newton-Schulz orthogonalization.
#[derive(Clone, Copy)]
struct NewtonSchulzParams {
    a: f32,
    b: f32,
    c: f32,
    steps: usize,
}

impl NewtonSchulzParams {
    fn new(coefficients: (f32, f32, f32), steps: usize) -> Self {
        Self {
            a: coefficients.0,
            b: coefficients.1,
            c: coefficients.2,
            steps,
        }
    }
}

/// Muon optimizer.
/// 
/// Muon internally runs standard SGD-momentum, and then performs an orthogonalization 
/// post-processing step, in which each 2D parameter's update is replaced with the 
/// nearest orthogonal matrix. For efficient orthogonalization we use a Newton-Schulz 
/// iteration, which has the advantage that it can be stably run in bfloat16 on the GPU.
#[derive(Clone)]
pub struct Muon<B: Backend> {
    momentum: Momentum<B>,
    ns_params: NewtonSchulzParams,
    weight_decay: Option<WeightDecay>,
    epsilon: f32,
}

impl<B: Backend> Muon<B> {
    /// Perform Newton-Schulz orthogonalization on a gradient tensor.
    /// 
    /// This computes the zeroth power (orthogonalization) of the input matrix G
    /// using a quintic Newton-Schulz iteration.
    /// 
    /// References:
    /// - original implementation: https://github.com/KellerJordan/Muon/blob/master/muon.py
    /// - pytorch implementation: https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py
    fn zeropower_via_newtonschulz<const D: usize>(
        &self,
        g: Tensor<B, D>,
    ) -> Tensor<B, D> {
        assert!(
            D >= 2,
            "Newton-Schulz iteration requires at least 2D tensors, got {}D",
            D
        );

        let shape = g.shape();
        let dims: [_; D] = shape.dims();
        let dim_m2 = dims[D - 2];
        let dim_m1 = dims[D - 1];

        // Step 1: Transpose if tall matrix
        let (mut x, needs_transpose) = if dim_m2 > dim_m1 {
            (g.swap_dims(D - 2, D - 1), true)
        } else {
            (g, false)
        };

        // Step 2: Normalize by Frobenius norm
        // X = X / (||X|| + epsilon)
        let norm = x.clone()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .add_scalar(self.epsilon)
            .into_scalar();
        
        x = x.div_scalar(norm);

        // Step 3: Newton-Schulz iteration
        let NewtonSchulzParams { a, b, c, steps } = self.ns_params;
        
        for _ in 0..steps {
            // A = X @ X^T
            // swap last two dims
            let x_t = x.clone().swap_dims(D - 2, D - 1);
            let a_matrix = x.clone().matmul(x_t);
            
            // B = b*A + c*A@A
            let a_squared = a_matrix.clone().matmul(a_matrix.clone());
            let b_matrix = a_matrix.mul_scalar(b).add(a_squared.mul_scalar(c));
            
            // X = a*X + B@X
            x = x.clone().mul_scalar(a).add(b_matrix.matmul(x.clone()));
        }

        // Step 4: Restore transpose
        if needs_transpose {
            x = x.swap_dims(D - 2, D - 1);
        }

        x
    }
}

/// Muon state.
#[derive(Record, Clone, new)]
pub struct MuonState<B: Backend, const D: usize> {
    pub momentum: MomentumState<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for Muon<B> {
    type State<const D: usize> = MuonState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        assert!(
            D >= 2,
            "Muon optimizer is designed for 2D+ parameters (matrices). \
            For 1D parameters (biases, layer norms), use AdamW or SGD instead."
        );

        let state_momentum = state.map(|s| s.momentum);
        
        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }
        
        let (grad, new_momentum_state) = self.momentum.transform(grad, state_momentum);
        
        let grad = self.zeropower_via_newtonschulz(grad);
        
        let delta = grad.mul_scalar(lr);
        let new_state = MuonState::new(new_momentum_state);
        
        (tensor - delta, Some(new_state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

impl MuonConfig {
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<Muon<B::InnerBackend>, M, B> {
        let momentum = Momentum::new(&self.momentum);
        let weight_decay = self.weight_decay.as_ref().map(WeightDecay::new);

        let optim = Muon {
            momentum,
            ns_params: NewtonSchulzParams::new(self.ns_coefficients, self.ns_steps),
            weight_decay,
            epsilon: self.epsilon,
        };

        OptimizerAdaptor::from(optim)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_muon_config_default() {
        let config = MuonConfig::new();
        assert_eq!(config.momentum.momentum, 0.95);
        assert_eq!(config.momentum.dampening, 0.0);
        assert_eq!(config.momentum.nesterov, true);
        assert!(config.weight_decay.is_none());
        assert_eq!(config.ns_steps, 5);
    }
    
    #[test]
    fn test_muon_config_builder_pattern() {
        let config = MuonConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(0.1)))
            .with_momentum(MomentumConfig {
                momentum: 0.9,
                dampening: 0.0,
                nesterov: false,
            })
            .with_ns_steps(7)
            .with_epsilon(1e-8);
        
        assert_eq!(config.momentum.momentum, 0.9);
        assert_eq!(config.momentum.nesterov, false);
        assert_eq!(config.ns_steps, 7);
        assert_eq!(config.epsilon, 1e-8);
    }
}
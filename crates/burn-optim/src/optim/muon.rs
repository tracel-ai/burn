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
    /// Weight decay (L2 penalty).
    /// 
    /// Uses AdamW-style decoupled weight decay.
    /// Default: 0.1
    #[config(default = 0.1)]
    weight_decay: f64,
    
    /// Momentum factor.
    /// 
    /// Coefficient for the moving average of gradients.
    /// Default: 0.95
    #[config(default = 0.95)]
    momentum: f64,
    
    /// Enable Nesterov momentum.
    /// 
    /// When true, uses Nesterov-style momentum which empirically
    /// works better than standard SGD momentum for Muon.
    /// Only applicable when momentum is non-zero.
    /// Default: true
    #[config(default = true)]
    nesterov: bool,
    
    /// Newton-Schulz iteration coefficients (a, b, c).
    /// 
    /// These coefficients define the quintic polynomial used in
    /// Newton-Schulz orthogonalization: f(X) = aX + bX(X^T X) + cX(X^T X)^2
    /// 
    /// The default coefficients (3.4445, -4.775, 2.0315) are optimized
    /// to maximize convergence speed while maintaining stability.
    /// Default: (3.4445, -4.775, 2.0315)
    #[config(default = "(3.4445, -4.775, 2.0315)")]
    ns_coefficients: (f64, f64, f64),
    
    /// Epsilon for numerical stability.
    /// 
    /// Small constant added to denominators to prevent division by zero
    /// when normalizing the spectral norm.
    /// Default: 1e-7
    #[config(default = 1e-7)]
    epsilon: f64,
    
    /// Number of Newton-Schulz iteration steps.
    /// 
    /// More steps = more accurate orthogonalization but higher computational cost.
    /// 5 steps is typically sufficient for good convergence.
    /// Default: 5
    #[config(default = 5)]
    ns_steps: usize,
}

/// Parameters for Newton-Schulz orthogonalization.
#[derive(Clone, Copy)]
struct NewtonSchulzParams {
    /// Coefficient 'a' in the quintic iteration.
    a: f32,
    
    /// Coefficient 'b' in the quintic iteration.
    b: f32,
    
    /// Coefficient 'c' in the quintic iteration.
    c: f32,
    
    /// Number of iteration steps.
    steps: usize,
}

impl NewtonSchulzParams {
    fn new(coefficients: (f64, f64, f64), steps: usize) -> Self {
        Self {
            a: coefficients.0 as f32,
            b: coefficients.1 as f32,
            c: coefficients.2 as f32,
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
    /// Momentum transformation (from SGD).
    momentum: Momentum<B>,
    
    /// Newton-Schulz iteration parameters.
    ns_params: NewtonSchulzParams,
    
    /// Weight decay transformation.
    weight_decay: Option<WeightDecay>,
    
    /// Epsilon for numerical stability.
    epsilon: f32,
}

impl<B: Backend> Muon<B> {
    /// Perform Newton-Schulz orthogonalization on a gradient tensor.
    /// 
    /// This computes the zeroth power (orthogonalization) of the input matrix G
    /// using a quintic Newton-Schulz iteration.
    /// 
    /// # Algorithm
    /// 
    /// ```text
    /// X = G / ||G||  (normalize spectral norm)
    /// for _ in range(steps):
    ///     A = X @ X^T
    ///     B = b*A + c*A@A
    ///     X = a*X + B@X
    /// ```
    /// 
    /// The iteration does not converge all the way to an orthogonal matrix, but rather
    /// produces something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
    /// which empirically does not hurt model performance relative to UV^T.
    fn zeropower_via_newtonschulz<const D: usize>(
        &self,
        g: Tensor<B, D>,
    ) -> Tensor<B, D> {
        // TODO: Implement Newton-Schulz iteration
        // 1. Handle transpose for tall matrices (rows > cols)
        // 2. Normalize spectral norm: X = G / (||G|| + epsilon)
        // 3. Perform quintic iteration
        // 4. Restore transpose if needed
        unimplemented!()
    }
}

/// Muon state.
#[derive(Record, Clone, new)]
pub struct MuonState<B: Backend, const D: usize> {
    /// Momentum state (if momentum is enabled).
    /// This reuses the MomentumState from SGD.
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
        // Muon requires 2D+ parameters (weight matrices)
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
        
        grad = self.zeropower_via_newtonschulz(grad);
        
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
    /// Initialize Muon optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self
    ) -> OptimizerAdaptor<Muon<B::InnerBackend>, M, B> {
        // When momentum=0, it behaves like vanilla gradient descent
        // but still maintains the momentum buffer
        let momentum = Momentum::new(&MomentumConfig {
            momentum: self.momentum,
            dampening: 0.0,  // No dampening for Muon
            nesterov: self.nesterov,
        });
        
        let weight_decay = if self.weight_decay > 0.0 {
            Some(WeightDecay::new(&WeightDecayConfig {
                penalty: self.weight_decay as f32,
            }))
        } else {
            None
        };

        let optim = Muon {
            momentum,
            ns_params: NewtonSchulzParams::new(self.ns_coefficients, self.ns_steps),
            weight_decay,
            epsilon: self.epsilon as f32,
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
        assert_eq!(config.momentum, 0.95);
        assert_eq!(config.weight_decay, 0.1);
        assert_eq!(config.nesterov, true);
        assert_eq!(config.ns_steps, 5);
    }   
}
use burn_core as burn;

use burn::{module::AutodiffModule, record::Record};

use burn::config::Config;
use burn::tensor::{Tensor, backend::AutodiffBackend};
use burn::tensor::{backend::Backend, ops::Device};

use super::{
    SimpleOptimizer,
    adaptor::OptimizerAdaptor,
    decay::WeightDecayConfig,
    momentum::{Momentum, MomentumConfig, MomentumState},
};
use crate::LearningRate;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Learning rate adjustment method for Muon optimizer.
///
/// Muon adjusts the learning rate based on parameter shape to maintain consistent
/// RMS across rectangular matrices.
///
/// # References
///
/// - Original: [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/)
/// - Moonshot: [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982)
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum AdjustLrFn {
    /// Keller Jordan's original method: `lr * sqrt(max(1, A/B))`
    ///
    /// This scales the learning rate based on the aspect ratio of the weight matrix,
    /// ensuring that tall matrices (more rows than columns) get proportionally larger
    /// learning rates.
    ///
    /// # Example
    ///
    /// For a [1024, 512] matrix: `lr * sqrt(1024/512) = lr * 1.414`
    Original,

    /// Moonshot's method: `lr * 0.2 * sqrt(max(A, B))`
    ///
    /// This method is designed to match AdamW's RMS, allowing Muon to directly reuse
    /// learning rates and weight decay values tuned for AdamW without retuning.
    ///
    /// # Example
    ///
    /// For a [1024, 512] matrix: `lr * 0.2 * sqrt(1024) = lr * 6.4`
    MatchRmsAdamW,
}

impl Default for AdjustLrFn {
    fn default() -> Self {
        Self::Original
    }
}

impl AdjustLrFn {
    /// Calculate the learning rate adjustment ratio for a given parameter shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - Parameter shape (uses first two dimensions)
    ///
    /// # Returns
    ///
    /// Adjustment ratio to multiply with the base learning rate
    fn adjustment_ratio(&self, shape: &[usize]) -> f64 {
        if shape.len() < 2 {
            return 1.0;
        }

        let a = shape[0] as f64;
        let b = shape[1] as f64;

        match self {
            Self::Original => {
                // sqrt(max(1, A/B))
                let ratio = a / b;
                ratio.max(1.0).sqrt()
            }
            Self::MatchRmsAdamW => {
                // 0.2 * sqrt(max(A, B))
                0.2 * a.max(b).sqrt()
            }
        }
    }
}

/// Muon configuration.
///
/// Muon is an optimizer specifically designed for 2D parameters of neural network
/// hidden layers (weight matrices). Other parameters such as biases and embeddings
/// should be optimized using a standard method such as AdamW.
///
/// # Learning Rate Adjustment
///
/// Muon adjusts the learning rate based on parameter shape to maintain consistent
/// RMS across rectangular matrices. Two methods are available:
///
/// - **Original**: Uses `sqrt(max(1, A/B))` where A and B are the first two dimensions.
///   This is Keller Jordan's method and is the default.
///
/// - **MatchRmsAdamW**: Uses `0.2 * sqrt(max(A, B))`. This is Moonshot's method
///   designed to match AdamW's RMS, allowing direct reuse of AdamW hyperparameters.
///
/// # Example
///
/// ```ignore
/// use burn_optim::{MuonConfig, AdjustLrFn};
///
/// // Using default (Original) method
/// let optimizer = MuonConfig::new().init();
///
/// // Using MatchRmsAdamW for AdamW-compatible hyperparameters
/// let optimizer = MuonConfig::new()
///     .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
///     .init();
/// ```
///
/// # References
///
/// - [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
/// - [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982)
/// - [PyTorch Implementation](https://github.com/pytorch/pytorch/blob/main/torch/optim/muon.py)
/// - [Original Implementation](https://github.com/KellerJordan/Muon)
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
    #[config(default = "MomentumConfig { momentum: 0.95, dampening: 0.0, nesterov: true }")]
    momentum: MomentumConfig,

    /// Newton-Schulz iteration coefficients (a, b, c).
    ///
    /// These coefficients are selected to maximize the slope at zero for the
    /// quintic iteration. Default values are from Keller Jordan's implementation.
    #[config(default = "(3.4445, -4.775, 2.0315)")]
    ns_coefficients: (f32, f32, f32),

    /// Epsilon for numerical stability.
    #[config(default = 1e-7)]
    epsilon: f32,

    /// Number of Newton-Schulz iteration steps.
    #[config(default = 5)]
    ns_steps: usize,

    /// Learning rate adjustment method.
    ///
    /// Controls how the learning rate is adjusted based on parameter shape.
    /// See [`AdjustLrFn`] for available methods.
    #[config(default = "AdjustLrFn::Original")]
    adjust_lr_fn: AdjustLrFn,
}

impl MuonConfig {
    /// Initialize Muon optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer adaptor that can be used to optimize a module.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use burn_optim::{MuonConfig, AdjustLrFn, decay::WeightDecayConfig};
    ///
    /// // Basic configuration with default (Original) LR adjustment
    /// let optimizer = MuonConfig::new()
    ///     .with_weight_decay(Some(WeightDecayConfig::new(0.01)))
    ///     .init();
    ///
    /// // With AdamW-compatible settings using MatchRmsAdamW
    /// let optimizer = MuonConfig::new()
    ///     .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
    ///     .with_weight_decay(Some(WeightDecayConfig::new(0.1)))
    ///     .init();
    ///
    /// // Custom momentum and NS settings
    /// let optimizer = MuonConfig::new()
    ///     .with_momentum(MomentumConfig {
    ///         momentum: 0.9,
    ///         dampening: 0.1,
    ///         nesterov: false,
    ///     })
    ///     .with_ns_steps(7)
    ///     .init();
    /// ```
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<Muon<B::InnerBackend>, M, B> {
        let momentum = Momentum::new(&self.momentum);
        let weight_decay_penalty = self.weight_decay.as_ref().map(|wd| wd.penalty);

        let optim = Muon {
            momentum,
            ns_params: NewtonSchulzParams::new(self.ns_coefficients, self.ns_steps),
            weight_decay_penalty,
            epsilon: self.epsilon,
            adjust_lr_fn: self.adjust_lr_fn,
        };

        OptimizerAdaptor::from(optim)
    }
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
///
/// # Important Notes
///
/// 1. **Only for 2D+ parameters**: Muon is designed for weight matrices. Use AdamW
///    or SGD for biases, embeddings, and layer norms.
///
/// 2. **Learning rate adjustment**: Muon automatically adjusts the learning rate based
///    on parameter shape. See [`AdjustLrFn`] for details.
///
/// 3. **Weight decay timing**: Unlike typical optimizers, Muon applies weight decay
///    AFTER orthogonalization but uses the original (unadjusted) learning rate for it.
#[derive(Clone)]
pub struct Muon<B: Backend> {
    momentum: Momentum<B>,
    ns_params: NewtonSchulzParams,
    weight_decay_penalty: Option<f32>,
    epsilon: f32,
    adjust_lr_fn: AdjustLrFn,
}

impl<B: Backend> Muon<B> {
    /// Adjust learning rate based on parameter shape.
    ///
    /// # Arguments
    ///
    /// * `lr` - Base learning rate
    /// * `shape` - Parameter shape (uses first two dimensions)
    ///
    /// # Returns
    ///
    /// Adjusted learning rate
    ///
    /// ```ignore
    /// // For a [1024, 512] weight matrix with lr=0.01:
    /// // Original: 0.01 * sqrt(1024/512) = 0.01 * 1.414 = 0.01414
    /// // MatchRmsAdamW: 0.01 * 0.2 * sqrt(1024) = 0.01 * 0.2 * 32 = 0.064
    /// ```
    fn adjust_lr(&self, lr: LearningRate, shape: &[usize]) -> LearningRate {
        lr * self.adjust_lr_fn.adjustment_ratio(shape)
    }

    /// Perform Newton-Schulz orthogonalization on a gradient tensor.
    ///
    /// This computes the zeroth power (orthogonalization) of the input matrix G
    /// using a quintic Newton-Schulz iteration.
    ///
    /// # Algorithm
    ///
    /// 1. Transpose if tall matrix (A > B)
    /// 2. Normalize: X = X / ||X||
    /// 3. For k steps:
    ///    - A = X @ X^T
    ///    - B = b*A + c*A^2
    ///    - X = a*X + B@X
    /// 4. Transpose back if needed
    ///
    /// # References
    ///
    /// - Original: https://github.com/KellerJordan/Muon/blob/master/muon.py
    /// - PyTorch: https://github.com/pytorch/pytorch/blob/main/torch/optim/muon.py
    fn zeropower_via_newtonschulz<const D: usize>(&self, g: Tensor<B, D>) -> Tensor<B, D> {
        assert!(
            D >= 2,
            "Newton-Schulz iteration requires at least 2D tensors, got {}D",
            D
        );

        let shape = g.shape();
        let dims: Vec<usize> = shape.dims::<D>().into();
        let dim_m2 = dims[D - 2];
        let dim_m1 = dims[D - 1];

        // Step 1: Transpose if tall matrix (more rows than columns)
        let (mut x, needs_transpose) = if dim_m2 > dim_m1 {
            (g.swap_dims(D - 2, D - 1), true)
        } else {
            (g, false)
        };

        // Step 2: Normalize by Frobenius norm
        // X = X / (||X|| + epsilon)
        let norm = x
            .clone()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .add_scalar(self.epsilon)
            .into_scalar();

        x = x.div_scalar(norm);

        // Step 3: Newton-Schulz iteration
        // This is the quintic iteration with coefficients (a, b, c)
        let NewtonSchulzParams { a, b, c, steps } = self.ns_params;

        for _ in 0..steps {
            // A = X @ X^T
            let x_t = x.clone().swap_dims(D - 2, D - 1);
            let a_matrix = x.clone().matmul(x_t);

            // B = b*A + c*A@A
            let a_squared = a_matrix.clone().matmul(a_matrix.clone());
            let b_matrix = a_matrix.mul_scalar(b).add(a_squared.mul_scalar(c));

            // X = a*X + B@X
            x = x.clone().mul_scalar(a).add(b_matrix.matmul(x.clone()));
        }

        // Step 4: Restore transpose if it was a tall matrix
        if needs_transpose {
            x = x.swap_dims(D - 2, D - 1);
        }

        x
    }
}

/// Muon state.
#[derive(Record, Clone, new)]
pub struct MuonState<B: Backend, const D: usize> {
    /// Current momentum state
    pub momentum: MomentumState<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for Muon<B> {
    type State<const D: usize> = MuonState<B, D>;

    /// Perform a single Muon optimization step.
    ///
    /// # Algorithm
    ///
    /// 1. Apply momentum to gradient
    /// 2. Orthogonalize update via Newton-Schulz
    /// 3. Adjust learning rate based on parameter shape
    /// 4. Apply weight decay (using original lr)
    /// 5. Update parameter (using adjusted lr)
    ///
    /// # Notes
    ///
    /// Unlike typical optimizers, the weight decay and parameter update use
    /// different learning rates:
    /// - Weight decay uses the original `lr`
    /// - Parameter update uses the shape-adjusted `lr`
    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        assert!(
            D >= 2,
            "Muon optimizer is designed for 2D+ parameters (matrices). \
            For 1D parameters (biases, layer norms), use AdamW or SGD instead. \
            Got {}D tensor.",
            D
        );

        // Step 1: Apply momentum
        let state_momentum = state.map(|s| s.momentum);
        let (grad, new_momentum_state) = self.momentum.transform(grad, state_momentum);

        // Step 2: Orthogonalize via Newton-Schulz
        let update = self.zeropower_via_newtonschulz(grad);

        // Step 3: Adjust learning rate based on parameter shape
        let shape_dims: [_; D] = tensor.shape().dims();
        let adjusted_lr = self.adjust_lr(lr, &shape_dims);

        // Step 4: Apply weight decay (using ORIGINAL lr, not adjusted)
        // Muon applies weight decay AFTER orthogonalization
        let tensor = if let Some(penalty) = self.weight_decay_penalty {
            let decay_factor = 1.0 - lr * penalty as f64;
            tensor.mul_scalar(decay_factor)
        } else {
            tensor
        };

        // Step 5: Update parameter (using ADJUSTED lr)
        let delta = update.mul_scalar(adjusted_lr);
        let new_state = MuonState::new(new_momentum_state);

        (tensor - delta, Some(new_state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAutodiffBackend;
    use crate::{GradientsParams, Optimizer};
    use burn::module::{Module, Param};
    use burn::tensor::{Distribution, Shape, Tensor, TensorData};
    use burn_nn::{Linear, LinearConfig, LinearRecord};

    type TestBackend = burn_ndarray::NdArray<f32>;

    fn given_linear_layer(weight: TensorData, bias: TensorData) -> Linear<TestAutodiffBackend> {
        let device = Default::default();
        let record = LinearRecord {
            weight: Param::from_data(weight, &device),
            bias: Some(Param::from_data(bias, &device)),
        };

        LinearConfig::new(4, 4).init(&device).load_record(record)
    }

    fn given_linear_layer_no_bias(weight: TensorData) -> Linear<TestAutodiffBackend> {
        let device = Default::default();
        let record = LinearRecord {
            weight: Param::from_data(weight, &device),
            bias: None, //No bias for Muon optimizer
        };

        LinearConfig::new(4, 4)
            .with_bias(false)
            .init(&device)
            .load_record(record)
    }

    #[test]
    fn test_adjust_lr_fn_enum_default() {
        let default = AdjustLrFn::default();
        assert_eq!(default, AdjustLrFn::Original);
    }

    #[test]
    fn test_adjust_lr_fn_original() {
        let method = AdjustLrFn::Original;

        // Square matrix [512, 512] -> sqrt(1) = 1.0
        let ratio = method.adjustment_ratio(&[512, 512]);
        assert!((ratio - 1.0).abs() < 1e-9);

        // Tall matrix [1024, 512] -> sqrt(2) ≈ 1.414
        let ratio = method.adjustment_ratio(&[1024, 512]);
        let expected = (2.0f64).sqrt();
        assert!((ratio - expected).abs() < 1e-9);

        // Wide matrix [512, 1024] -> max(1, 0.5) = 1.0
        let ratio = method.adjustment_ratio(&[512, 1024]);
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adjust_lr_fn_match_rms_adamw() {
        let method = AdjustLrFn::MatchRmsAdamW;

        // [1024, 512] -> 0.2 * sqrt(1024) = 6.4
        let ratio = method.adjustment_ratio(&[1024, 512]);
        let expected = 0.2 * 1024.0f64.sqrt();
        assert!((ratio - expected).abs() < 1e-9);

        // [512, 512] -> 0.2 * sqrt(512) ≈ 4.525
        let ratio = method.adjustment_ratio(&[512, 512]);
        let expected = 0.2 * 512.0f64.sqrt();
        assert!((ratio - expected).abs() < 1e-9);
    }

    #[test]
    fn test_lr_adjustment_with_enum() {
        let config_original = MuonConfig::new().with_adjust_lr_fn(AdjustLrFn::Original);

        let config_match = MuonConfig::new().with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW);

        assert_eq!(config_original.adjust_lr_fn, AdjustLrFn::Original);
        assert_eq!(config_match.adjust_lr_fn, AdjustLrFn::MatchRmsAdamW);
    }

    #[test]
    fn test_muon_with_original_method() {
        let config = MuonConfig::new().with_adjust_lr_fn(AdjustLrFn::Original);

        let muon: Muon<TestBackend> = Muon {
            momentum: Momentum::new(&config.momentum),
            ns_params: NewtonSchulzParams::new(config.ns_coefficients, config.ns_steps),
            weight_decay_penalty: None,
            epsilon: config.epsilon,
            adjust_lr_fn: config.adjust_lr_fn,
        };

        let lr = 0.01;
        let shape = vec![1024, 512];
        let adjusted = muon.adjust_lr(lr, &shape);
        let expected = lr * (2.0f64).sqrt();

        assert!((adjusted - expected).abs() < 1e-9);
    }

    #[test]
    fn test_muon_with_match_rms_adamw() {
        let config = MuonConfig::new().with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW);

        let muon: Muon<TestBackend> = Muon {
            momentum: Momentum::new(&config.momentum),
            ns_params: NewtonSchulzParams::new(config.ns_coefficients, config.ns_steps),
            weight_decay_penalty: None,
            epsilon: config.epsilon,
            adjust_lr_fn: config.adjust_lr_fn,
        };

        let lr = 0.01;
        let shape = vec![1024, 512];
        let adjusted = muon.adjust_lr(lr, &shape);
        let expected = lr * 0.2 * 1024.0f64.sqrt();

        assert!((adjusted - expected).abs() < 1e-9);
    }

    #[test]
    fn test_config_builder_with_enum() {
        // Test that the builder pattern works correctly with enum
        let optimizer = MuonConfig::new()
            .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
            .with_weight_decay(Some(WeightDecayConfig::new(0.1)))
            .with_ns_steps(7);

        assert_eq!(optimizer.adjust_lr_fn, AdjustLrFn::MatchRmsAdamW);
        assert_eq!(optimizer.ns_steps, 7);
    }

    #[test]
    #[should_panic(expected = "2D+ parameters")]
    fn test_1d_tensor_panics() {
        let device = Default::default();
        let config = MuonConfig::new();
        let optim: Muon<TestBackend> = Muon {
            momentum: Momentum::new(&config.momentum),
            ns_params: NewtonSchulzParams::new(config.ns_coefficients, config.ns_steps),
            weight_decay_penalty: None,
            epsilon: config.epsilon,
            adjust_lr_fn: config.adjust_lr_fn,
        };

        let tensor_1d = Tensor::<TestBackend, 1>::zeros([512], &device);
        let grad_1d = Tensor::<TestBackend, 1>::ones([512], &device);

        let _ = optim.step(0.01, tensor_1d, grad_1d, None);
    }

    #[test]
    fn test_muon_optimizer_save_load_state() {
        let device = Default::default();
        // Use Linear layer WITHOUT bias for Muon optimizer
        let linear = LinearConfig::new(6, 6)
            .with_bias(false) // No bias - only 2D weight matrix
            .init::<TestAutodiffBackend>(&device);

        let x = Tensor::<TestAutodiffBackend, 2>::random([2, 6], Distribution::Default, &device);

        let mut optimizer =
            MuonConfig::new().init::<TestAutodiffBackend, Linear<TestAutodiffBackend>>();
        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let _linear = optimizer.step(0.01, linear, grads);

        let state_before = optimizer.to_record();
        let state_before_copy = optimizer.to_record();

        let optimizer_new =
            MuonConfig::new().init::<TestAutodiffBackend, Linear<TestAutodiffBackend>>();
        let optimizer_loaded = optimizer_new.load_record(state_before_copy);
        let state_after = optimizer_loaded.to_record();

        assert_eq!(state_before.len(), state_after.len());
    }

    #[test]
    fn test_muon_with_weight_decay() {
        let device = Default::default();
        // Create Linear layer WITHOUT bias for Muon
        let linear = given_linear_layer_no_bias(TensorData::from([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]));

        let x = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
            &device,
        )
        .require_grad();

        let mut optimizer = MuonConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(0.01)))
            .init::<TestAutodiffBackend, Linear<TestAutodiffBackend>>();

        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(0.01, linear, grads);

        let state = linear.into_record();
        let weight = state.weight.to_data();

        for val in weight.as_slice::<f32>().unwrap() {
            assert!(
                *val < 1.0,
                "Weight should be reduced by weight decay, got {}",
                val
            );
        }
    }

    #[test]
    fn test_newton_schulz_orthogonalization() {
        let device = Default::default();
        let matrix = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.5], [0.5, 1.0]], &device);

        let config = MuonConfig::new();
        let muon: Muon<TestBackend> = Muon {
            momentum: Momentum::new(&config.momentum),
            ns_params: NewtonSchulzParams::new(config.ns_coefficients, config.ns_steps),
            weight_decay_penalty: None,
            epsilon: config.epsilon,
            adjust_lr_fn: config.adjust_lr_fn,
        };

        let orthogonalized = muon.zeropower_via_newtonschulz(matrix);
        let o_t = orthogonalized.clone().transpose();
        let product = orthogonalized.matmul(o_t);

        let data = product.into_data();
        let values = data.as_slice::<f32>().unwrap();

        assert!(
            (values[0] - 1.0).abs() < 0.1,
            "Product[0,0] should be ~1.0, got {}",
            values[0]
        );
        assert!(
            (values[3] - 1.0).abs() < 0.1,
            "Product[1,1] should be ~1.0, got {}",
            values[3]
        );
    }
}

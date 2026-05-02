//! An 8-bit optimizer of AdamW.

use burn_core as burn;

use burn::config::Config;
use burn::tensor::{
    Tensor,
    backend::{AutodiffBackend, Backend},
    ops::Device,
};
use burn::{module::AutodiffModule, record::Record};

use super::{SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::quantization::{
    QuantizeBlockwise, dequantize_blockwise, quantize_blockwise, signed_dynamic, unsigned_dynamic,
};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// [`AdamW8BitFused`] Configuration.
#[derive(Config, Debug)]
pub struct AdamWConfig8BitFused {
    /// Parameter for AdamW.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Parameter for AdamW.
    #[config(default = 0.999)]
    beta_2: f32,
    /// The amount of quantization applied to the optimizer. Always use a power of 2, or have
    /// highly degraded performance. Default value for the bitsandbytes library.
    #[config(default = 256)]
    block_size: usize,
    /// A value required for numerical stability.
    #[config(default = 1e-5)]
    epsilon: f32,
    /// Weight decay config.
    #[config(default = 1e-4)]
    weight_decay: f32,

    /// Cautious weight decay config.
    ///
    /// See: <https://arxiv.org/abs/2510.12402>
    #[config(default = false)]
    cautious_weight_decay: bool,

    /// Whether to use AMSGrad algorithm
    #[config(default = false)]
    amsgrad: bool,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,
}

/// AdamW 8 bit optimizer.
///
/// See:
/// - [`crate::optim::AdamW`] module
/// - [8-Bit Optimizers via Block-Wise Quantization](https://arxiv.org/pdf/2110.02861)
///
/// Configured by [`AdamWConfig`].
#[derive(Clone)]
pub struct AdamW8BitFused {
    momentum: AdaptiveMomentumW8Bit,
    weight_decay: f32,
    cautious_weight_decay: bool,
}

/// [`AdamW8Bit`] state.
#[derive(Record, Clone)]
pub struct AdamWState8BitFused<B: Backend, const D: usize> {
    time: usize,
    moment_1: QuantizeBlockwise<B, D>,
    moment_2: QuantizeBlockwise<B, D>,
    max_moment_2: Option<QuantizeBlockwise<B, D>>,
}

impl<B: Backend> SimpleOptimizer<B> for AdamW8BitFused {
    type State<const D: usize> = AdamWState8BitFused<B, D>;

    /// A single optimization step for any tensor that represents the parameters of a model.
    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let (raw_delta, new_state, m1) = self.momentum.transform(grad, state);

        let decay_rate = lr * (self.weight_decay as f64);
        let decayed_tensor = if decay_rate == 0.0 {
            tensor.clone()
        } else if self.cautious_weight_decay {
            let tensor_pos = tensor.clone().greater_equal_elem(0.0);

            let grad_pos = m1.greater_equal_elem(0.0);
            let differ = tensor_pos.not_equal(grad_pos);

            let decay = tensor.clone().mul_scalar(decay_rate).mask_fill(differ, 0.0);
            tensor.clone() - decay
        } else {
            tensor.clone().mul_scalar(1.0 - decay_rate)
        };

        let tensor_updated = decayed_tensor - raw_delta.mul_scalar(lr);

        (tensor_updated, Some(new_state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.moment_1 = state.moment_1.to_device(device);
        state.moment_2 = state.moment_2.to_device(device);
        state.max_moment_2 = state.max_moment_2.map(|m| m.to_device(device));
        state
    }
}

impl AdamWConfig8BitFused {
    /// Initialize [`AdamW8Bit`] optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<AdamW8BitFused, M, B> {
        let optim = AdamW8BitFused {
            momentum: AdaptiveMomentumW8Bit {
                beta_1: self.beta_1,
                beta_2: self.beta_2,
                epsilon: self.epsilon,
                amsgrad: self.amsgrad,
                block_size: self.block_size,
            },
            weight_decay: self.weight_decay,
            cautious_weight_decay: self.cautious_weight_decay,
        };

        let mut optim = OptimizerAdaptor::from(optim);
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

#[derive(Clone)]
struct AdaptiveMomentumW8Bit {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    amsgrad: bool,
    block_size: usize,
}

impl AdaptiveMomentumW8Bit {
    pub fn transform<B, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        state: Option<AdamWState8BitFused<B, D>>,
    ) -> (Tensor<B, D>, AdamWState8BitFused<B, D>, Tensor<B, D>)
    where
        B: CubeBackend,
    {
        // ---- Compute scalar bias corrections on the host ----
        let time = state.as_ref().map(|s| s.time + 1).unwrap_or(1);
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;
        let correction1 = 1.0 - self.beta_1.powi(time as i32);
        let correction2 = (1.0 - self.beta_2.powi(time as i32)).sqrt();

        let is_first_step = state.is_none();
        let device = grad.device();
        let shape = grad.shape();

        // ---- Get input state tensors (or zero placeholders on the first step) ----
        let (m1_in, m2_in, max_v_in) = match state {
            Some(s) => (Some(s.moment_1), Some(s.moment_2), s.max_moment_2),
            None => (None, None, None),
        };

        // ---- Launch the fused kernel via the launch wrapper ----
        let outputs = kernel::launch_adamw_8bit_transform::<B, D>(
            &grad,
            m1_in.as_ref(),
            m2_in.as_ref(),
            max_v_in.as_ref(),
            kernel::TransformParams {
                beta_1: self.beta_1,
                beta_2: self.beta_2,
                factor_1,
                factor_2,
                correction1,
                correction2,
                epsilon: self.epsilon,
                block_size: self.block_size as u32,
                amsgrad: self.amsgrad,
                is_first_step,
            },
        );

        // ---- Repack outputs into the typed state ----
        let new_state = AdamWState8BitFused {
            time,
            moment_1: outputs.moment_1_new,
            moment_2: outputs.moment_2_new,
            max_moment_2: outputs.max_moment_2_new,
        };

        (outputs.update_delta, new_state, outputs.m1_dequantized)
    }
}

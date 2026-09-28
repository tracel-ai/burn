use burn_core as burn;

use burn::{
    config::Config,
    tensor::{DType, Device, FloatDType, Tensor},
};

use crate::{LearningRate, RecordState, grad_clipping::GradientClippingConfig};

use super::{ModuleOptimizer, Optimizer};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// [`Adafactor`] configuration.
///
/// By default, the supplied learning rate caps the relative step size at
/// `min(lr, 1 / sqrt(step))`, which is then multiplied by the parameter's root mean square
/// (RMS). A learning rate of `0.01` gives the schedule proposed in the paper.
/// Disable both `relative_step` and `scale_parameter` to use an absolute learning rate
/// controlled entirely by an external scheduler.
#[derive(Config, Debug)]
pub struct AdafactorConfig {
    /// Positive constant added to squared gradients for numerical stability.
    #[config(default = 1e-30)]
    epsilon_1: f32,
    /// Positive lower bound on the parameter RMS when scaling the step size.
    #[config(default = 1e-3)]
    epsilon_2: f32,
    /// Positive upper bound on the RMS of the normalized update before learning rate scaling.
    /// This is separate from gradient clipping.
    #[config(default = 1.0)]
    clip_threshold: f32,
    /// Negative exponent in the second moment decay schedule `beta_2 = 1 - step^decay_rate`.
    #[config(default = -0.8)]
    decay_rate: f64,
    /// Whether to cap the supplied learning rate by `1 / sqrt(step)`.
    #[config(default = true)]
    relative_step: bool,
    /// Whether to multiply the step size by `max(epsilon_2, RMS(parameter))`.
    #[config(default = true)]
    scale_parameter: bool,
    /// Decoupled weight decay coefficient, multiplied by the supplied learning rate.
    #[config(default = 0.0)]
    weight_decay: f32,
    /// Optional gradient clipping applied before the Adafactor update.
    grad_clipping: Option<GradientClippingConfig>,
}

/// Adafactor optimizer with factored second moments and update clipping.
///
/// Matrices store only row and column second moments. Higher-rank tensors factor their
/// last two dimensions independently for each leading index; vectors store a full second
/// moment. No first moment is stored. Half-precision parameters use `f32` updates and state.
///
/// See [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235).
/// Configured by [`AdafactorConfig`].
#[derive(Clone)]
pub struct Adafactor {
    epsilon_1: f32,
    epsilon_2: f32,
    clip_threshold: f32,
    decay_rate: f64,
    relative_step: bool,
    scale_parameter: bool,
    weight_decay: f32,
}

/// Adafactor state for a single parameter tensor.
#[derive(RecordState, Clone)]
pub struct AdafactorState<const D: usize> {
    /// Number of updates applied to this parameter.
    pub time: usize,
    /// Moving average of squared gradients, reduced over the last dimension for matrices
    /// and higher-rank tensors (shape `[..., rows, 1]`). Vectors retain their full shape.
    pub second_moment: Tensor<D>,
    /// Column second moment for matrices and higher-rank tensors (shape `[..., 1, columns]`).
    /// Absent for vectors.
    pub column_second_moment: Option<Tensor<D>>,
}

impl AdafactorConfig {
    /// Build the per-parameter Adafactor optimizer.
    ///
    /// Use [`Self::init`] to construct a whole-module optimizer with the configured gradient
    /// clipping, or pass this optimizer to [`ModuleOptimizer::with_group`].
    ///
    /// # Panics
    ///
    /// Panics if either epsilon or the clipping threshold is not positive and finite,
    /// the decay exponent is not negative and finite, or weight decay is not nonnegative
    /// and finite.
    pub fn build(&self) -> Adafactor {
        assert!(
            self.epsilon_1.is_finite() && self.epsilon_1 > 0.0,
            "Adafactor epsilon_1 must be positive and finite"
        );
        assert!(
            self.epsilon_2.is_finite() && self.epsilon_2 > 0.0,
            "Adafactor epsilon_2 must be positive and finite"
        );
        assert!(
            self.clip_threshold.is_finite() && self.clip_threshold > 0.0,
            "Adafactor clip_threshold must be positive and finite"
        );
        assert!(
            self.decay_rate.is_finite() && self.decay_rate < 0.0,
            "Adafactor decay_rate must be negative and finite"
        );
        assert!(
            self.weight_decay.is_finite() && self.weight_decay >= 0.0,
            "Adafactor weight_decay must be nonnegative and finite"
        );

        Adafactor {
            epsilon_1: self.epsilon_1,
            epsilon_2: self.epsilon_2,
            clip_threshold: self.clip_threshold,
            decay_rate: self.decay_rate,
            relative_step: self.relative_step,
            scale_parameter: self.scale_parameter,
            weight_decay: self.weight_decay,
        }
    }

    /// Initialize a whole-module Adafactor optimizer.
    pub fn init(&self) -> ModuleOptimizer {
        let mut optimizer = ModuleOptimizer::from(self.build());
        if let Some(config) = &self.grad_clipping {
            optimizer = optimizer.with_grad_clipping(config.init());
        }
        optimizer
    }
}

impl Optimizer for Adafactor {
    type State<const D: usize> = AdafactorState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let dtype = tensor.dtype();
        // Squaring and adding epsilon in half precision can underflow, even for ordinary
        // gradients. Keep the accumulated statistics and the whole update in f32.
        let (tensor, grad) = match dtype {
            DType::F16 | DType::BF16 => (tensor.cast(FloatDType::F32), grad.cast(FloatDType::F32)),
            _ => (tensor, grad),
        };

        let squared_grad = grad.clone().square().add_scalar(self.epsilon_1);
        let (second_moment, column_second_moment) = if D >= 2 {
            (
                squared_grad.clone().mean_dim(D - 1),
                Some(squared_grad.mean_dim(D - 2)),
            )
        } else {
            (squared_grad, None)
        };

        let state = match state {
            Some(mut state) => {
                state.time += 1;
                let new_weight = (state.time as f64).powf(self.decay_rate);
                let old_weight = 1.0 - new_weight;
                state.second_moment = state
                    .second_moment
                    .mul_scalar(old_weight)
                    .add(second_moment.mul_scalar(new_weight));
                state.column_second_moment = column_second_moment.map(|column| {
                    state
                        .column_second_moment
                        .take()
                        .expect("Factored Adafactor state must contain a column second moment")
                        .mul_scalar(old_weight)
                        .add(column.mul_scalar(new_weight))
                });
                state
            }
            None => AdafactorState {
                time: 1,
                second_moment,
                column_second_moment,
            },
        };

        let mut update = match &state.column_second_moment {
            Some(column) => {
                let row = state.second_moment.clone();
                // Normalize the row factor first to avoid underflow from the product of
                // small moments. Broadcasting avoids storing a full second moment.
                let row_scale = row.clone().div(row.mean_dim(D - 2)).sqrt();
                grad.div(row_scale).div(column.clone().sqrt())
            }
            None => grad.div(state.second_moment.clone().sqrt()),
        };

        let clipping = rms(update.clone())
            .div_scalar(self.clip_threshold)
            .clamp_min(1.0);
        update = update.div(clipping.unsqueeze());

        let step_size = if self.relative_step {
            lr.min(1.0 / (state.time as f64).sqrt())
        } else {
            lr
        };
        if self.scale_parameter {
            let scale = rms(tensor.clone()).clamp_min(self.epsilon_2);
            update = update.mul(scale.unsqueeze());
        }
        update = update.mul_scalar(step_size);

        let tensor = if self.weight_decay == 0.0 {
            tensor
        } else {
            tensor.mul_scalar(1.0 - lr * self.weight_decay as f64)
        };
        let tensor = tensor - update;
        let tensor = match dtype {
            DType::F16 | DType::BF16 => tensor.cast(FloatDType::from(dtype)),
            _ => tensor,
        };

        (tensor, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.second_moment = state.second_moment.to_device(device);
        state.column_second_moment = state
            .column_second_moment
            .map(|column| column.to_device(device));
        state
    }
}

fn rms<const D: usize>(tensor: Tensor<D>) -> Tensor<1> {
    tensor.square().mean().sqrt()
}

#[cfg(test)]
mod tests;

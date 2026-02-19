use super::Reduction;
use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};
use burn_core as burn;

/// Configuration for the [SmoothL1Loss](SmoothL1Loss) module.
///
/// Smooth L1 loss combines L1 and L2 loss, using L2 loss for small errors (below beta)
/// and L1 loss for large errors (above beta). This makes it less sensitive to outliers
/// than MSE while maintaining smooth gradients near zero.
///
/// # Example
///
/// ```ignore
/// use burn_nn::loss::{SmoothL1LossConfig, Reduction};
///
/// // Create Smooth L1 loss with default beta=1.0
/// let smooth_l1 = SmoothL1LossConfig::new().init();
///
/// // Create with custom beta
/// let smooth_l1_custom = SmoothL1LossConfig::new().with_beta(0.5).init();
/// ```
#[derive(Config, Debug)]
pub struct SmoothL1LossConfig {
    /// Specifies the threshold at which to change between L1 and L2 loss.
    /// The value must be positive. Default: 1.0
    #[config(default = 1.0)]
    pub beta: f32,
}

impl SmoothL1LossConfig {
    /// Initializes a [Smooth L1 Loss](SmoothL1Loss) module.
    ///
    /// # Panics
    ///
    /// Panics if `beta <= 0`.
    pub fn init(&self) -> SmoothL1Loss {
        self.assertions();
        SmoothL1Loss { beta: self.beta }
    }

    fn assertions(&self) {
        assert!(self.beta > 0.0, "The parameter beta must be positive.")
    }
}

/// Computes the Smooth L1 Loss between predictions and targets.
///
/// This loss function uses L2 loss for small errors (below beta) and L1 loss for
/// large errors (above beta), providing robustness to outliers while maintaining
/// smooth gradients near |x - y| = 0.
///
/// # Mathematical Definition
///
/// For predictions `x` and targets `y`, the element-wise loss is:
///
/// - L_i = 0.5 * (x_i - y_i)² / beta   , if |x_i - y_i| < beta
/// - L_i = |x_i - y_i| - 0.5 * beta    , otherwise
///
/// # Notes
///
/// Smooth L1 loss is closely related to HuberLoss since it is equivalent to HuberLoss
/// scaled by `1/beta`:
/// `SmoothL1(x, y, beta) = Huber(x, y, beta) / beta`
///
/// This leads to the following differences:
///
/// - As beta approaches 0, Smooth L1 loss converges to L1Loss, while HuberLoss converges to 0.
///   When beta = 0, Smooth L1 loss is equivalent to L1 loss. Thus, the `beta`
///   parameter in Burn must be positive. L1Loss should be used for beta = 0.
/// - As beta approaches positive infinity, Smooth L1 loss converges to a constant 0 loss, while
///   HuberLoss converges to L2Loss.
///
/// # Example
///
/// ```rust,ignore
/// use burn_nn::loss::{SmoothL1LossConfig, Reduction};
/// use burn::tensor::Tensor;
///
/// // Create Smooth L1 loss with the default beta=1.0
/// let smooth_l1 = SmoothL1LossConfig::new().init();
///
/// let predictions: Tensor<Backend, 2> = /* model output */;
/// let targets: Tensor<Backend, 2> = /* ground truth */;
///
/// // Compute element-wise loss without reduction
/// let element_wise = smooth_l1.forward(predictions.clone(), targets.clone());
///
/// // Compute loss with mean reduction
/// let loss = smooth_l1.forward_with_reduction(predictions.clone(), targets.clone(), Reduction::Mean);
///
/// // Per-image loss: reduce over C, H, W → [batch, 1, 1, 1]
/// let loss_per_image = smooth_l1.forward_reduce_dims(predictions, targets, &[1, 2, 3]);
/// ```
#[derive(Module, Clone, Debug)]
pub struct SmoothL1Loss {
    /// Specifies the threshold at which to change between L1 and L2 loss.
    /// The value must be positive. Default: 1.0
    pub beta: f32,
}

impl SmoothL1Loss {
    /// Computes the element-wise smooth L1 loss without reduction.
    ///
    /// # Arguments
    ///
    /// - `predictions` - The model's predicted values.
    /// - `targets` - The ground truth target values.
    ///
    /// # Returns
    ///
    /// A tensor of the same shape as the inputs, containing the smooth L1 loss
    /// for each element.
    ///
    /// # Shapes
    ///
    /// - predictions: `[...dims]` - Any shape
    /// - targets: `[...dims]` - Must match predictions shape
    /// - output: `[...dims]` - Same shape as inputs
    pub fn forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let error = predictions.sub(targets);
        let abs_error = error.clone().abs();

        // The L1 case: |error| - 0.5 * beta (when |error| >= beta)
        let l1_loss = abs_error.clone().sub_scalar(0.5 * self.beta);

        // The L2 case: 0.5 * (error)^2 / beta (when |error| < beta)
        let l2_loss = error.square().mul_scalar(0.5).div_scalar(self.beta);

        let l2_mask = abs_error.lower_elem(self.beta);
        l1_loss.mask_where(l2_mask, l2_loss)
    }

    /// Computes the smooth L1 loss with reduction.
    ///
    /// # Arguments
    ///
    /// - `predictions` - The model's predicted values.
    /// - `targets` - The ground truth target values.
    /// - `reduction` - Specifies how to reduce the element-wise losses:
    ///   - `Reduction::Mean` or `Reduction::Auto`: Returns the mean of all element-wise losses.
    ///   - `Reduction::Sum`: Returns the sum of all element-wise losses.
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the reduced loss value.
    ///
    /// # Shapes
    ///
    /// - predictions: `[...dims]` - Any shape
    /// - targets: `[...dims]` - Must match predictions shape
    /// - output: `[1]` - Scalar loss value
    pub fn forward_with_reduction<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let unreduced_loss = self.forward(predictions, targets);

        match reduction {
            Reduction::Mean | Reduction::Auto => unreduced_loss.mean(),
            Reduction::Sum => unreduced_loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Computes the smooth L1 loss with reduction over specified dimensions.
    ///
    /// Calculates element-wise smooth L1 loss, then takes the mean
    /// over the specified dimensions. Useful for per-sample or per-channel losses.
    ///
    /// Dimensions can be provided in any order. They are sorted internally and
    /// reduced from highest to lowest to ensure indices remain valid.
    ///
    /// # Arguments
    ///
    /// - `predictions` - The model's predicted values.
    /// - `targets` - The ground truth target values.
    /// - `dims` - Dimensions to reduce over.
    ///
    /// # Returns
    ///
    /// A tensor with the specified dimensions reduced to size 1.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Consider image tensor with shape [batch, C, H, W]
    /// let smooth_l1 = SmoothL1LossConfig::new().init();
    ///
    /// // Per-image loss: reduce over C, H, W → [batch, 1, 1, 1]
    /// let loss_per_image = smooth_l1.forward_reduce_dims(predictions, targets, &[1, 2, 3]);
    /// ```
    pub fn forward_reduce_dims<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        dims: &[usize],
    ) -> Tensor<B, D> {
        let error = self.forward(predictions, targets);

        // Sort the dimensions to ascending order
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort();

        // Reduce over specified dimensions
        let mut result = error;
        for &dim in sorted_dims.iter().rev() {
            result = result.mean_dim(dim);
        }
        result
    }
}

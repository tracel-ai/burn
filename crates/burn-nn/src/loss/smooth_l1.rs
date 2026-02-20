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
        error.mean_dims(sorted_dims.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;
    use burn::tensor::{Tolerance, ops::FloatElem};

    type FT = FloatElem<TestBackend>;

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_smooth_l1_config_default_beta() {
        let loss = SmoothL1LossConfig::new().init();
        assert_eq!(loss.beta, 1.0);
    }

    #[test]
    fn test_smooth_l1_config_custom_beta() {
        let loss = SmoothL1LossConfig::new().with_beta(2.5).init();
        assert_eq!(loss.beta, 2.5);
    }

    #[test]
    #[should_panic(expected = "The parameter beta must be positive")]
    fn test_smooth_l1_config_beta_zero_panics() {
        SmoothL1LossConfig::new().with_beta(0.0).init();
    }

    #[test]
    #[should_panic(expected = "The parameter beta must be positive")]
    fn test_smooth_l1_config_beta_negative_panics() {
        SmoothL1LossConfig::new().with_beta(-1.0).init();
    }

    // =========================================================================
    // Forward Pass (Element-wise) Tests
    // =========================================================================

    #[test]
    fn test_smooth_l1_forward_l2_region() {
        // Beta = 1.0, errors = 0.0 and 0.5 (both < beta, use L2 formula)
        // L2 formula: 0.5 * error^2 / beta
        // error = 0.0  ->  loss = 0.5 * 0.0 / 1.0 = 0.0
        // error = 0.5  ->  loss = 0.5 * 0.25 / 1.0 = 0.125
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0_f32, 0.5]]), &device);
        let targets =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0_f32, 0.0]]), &device);

        let output = loss.forward(predictions, targets);
        let expected = TensorData::from([[0.0_f32, 0.125]]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_smooth_l1_forward_l1_region() {
        // Beta = 1.0, errors = 0.0 and 2.0 (2.0 >= beta, use L1 formula)
        // L1 formula: |error| - 0.5 * beta
        // L2 formula: 0.5 * (error)^2 / beta
        // error = 0.0  ->  loss = 0.0
        // error = 2.0  ->  loss = 2.0 - 0.5 = 1.5
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0_f32, 2.0]]), &device);
        let targets =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0_f32, 0.0]]), &device);

        let output = loss.forward(predictions, targets);
        let expected = TensorData::from([[0.0_f32, 1.5]]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_smooth_l1_forward_zero_error() {
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[1.0_f32, 2.0, 3.0]]), &device);
        let targets = predictions.clone();

        let output = loss.forward(predictions, targets);
        let expected = TensorData::from([[0.0_f32, 0.0, 0.0]]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_smooth_l1_forward_negative_errors() {
        // Ensure absolute value is used correctly
        // L1 formula: |error| - 0.5 * beta
        // L2 formula: 0.5 * (error)^2 / beta
        // Beta = 1.0, error = -3.0 (L1: 3.0 - 0.5 = 2.5)
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([-3.0_f32]), &device);
        let targets = Tensor::<TestBackend, 1>::zeros([1], &device);

        let output = loss.forward(predictions, targets);
        let expected = TensorData::from([2.5_f32]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_smooth_l1_forward_mixed_regions() {
        // Test with errors in both L1 and L2 regions
        // Beta = 1.0
        // L1 formula: |error| - 0.5 * beta
        // L2 formula: 0.5 * (error)^2 / beta
        // error = 0.5 -> L2: 0.5 * 0.25 / 1 = 0.125
        // error = 1.5 -> L1: 1.5 - 0.5 = 1.0
        // error = 3.0 -> L1: 3.0 - 0.5 = 2.5
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([0.5_f32, 1.5, 3.0]), &device);
        let targets = Tensor::<TestBackend, 1>::zeros([3], &device);

        let output = loss.forward(predictions, targets);
        let expected = TensorData::from([0.125_f32, 1.0, 2.5]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_smooth_l1_custom_beta_values() {
        // Test with beta = 0.5
        // error = 0.25 (< beta): L2 = 0.5 * 0.0625 / 0.5 = 0.0625
        // error = 1.0 (>= beta): L1 = 1.0 - 0.25 = 0.75
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().with_beta(0.5).init();

        let predictions =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([0.25_f32, 1.0]), &device);
        let targets = Tensor::<TestBackend, 1>::zeros([2], &device);

        let output = loss.forward(predictions, targets);
        let expected = TensorData::from([0.0625_f32, 0.75]);
        output.into_data().assert_eq(&expected, false);
    }

    // =========================================================================
    // forward_with_reduction Tests
    // =========================================================================

    #[test]
    fn test_smooth_l1_reduction_mean() {
        // Errors: 0.5 (L2: 0.125), 2.0 (L1: 1.5)
        // Mean: (0.125 + 1.5) / 2 = 0.8125
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.5_f32, 2.0]]), &device);
        let targets =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0_f32, 0.0]]), &device);

        let output = loss.forward_with_reduction(predictions, targets, Reduction::Mean);
        let expected = TensorData::from([0.8125_f32]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_smooth_l1_reduction_sum() {
        // Errors: 0.5 (L2: 0.125), 2.0 (L1: 1.5)
        // Sum: 1.625
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.5_f32, 2.0]]), &device);
        let targets =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0_f32, 0.0]]), &device);

        let output = loss.forward_with_reduction(predictions, targets, Reduction::Sum);
        let expected = TensorData::from([1.625_f32]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_smooth_l1_reduction_auto_equals_mean() {
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions = Tensor::<TestBackend, 1>::from_data(TensorData::from([2.0_f32]), &device);
        let targets = Tensor::<TestBackend, 1>::zeros([1], &device);

        let mean_out =
            loss.forward_with_reduction(predictions.clone(), targets.clone(), Reduction::Mean);
        let auto_out = loss.forward_with_reduction(predictions, targets, Reduction::Auto);

        mean_out.into_data().assert_eq(&auto_out.into_data(), false);
    }

    // =========================================================================
    // Dimension Reduction Tests
    // =========================================================================

    #[test]
    fn test_smooth_l1_forward_reduce_dims_single_dim() {
        // Beta = 2.0
        // L1 formula: |error| - 0.5 * beta
        // L2 formula: 0.5 * (error)^2 / beta
        // Row 0: errors [0.0, 1.0, 4.0]
        //   error = 0.0 -> L2: 0.0
        //   error = 1.0 -> L2: 0.5 * 1.0 / 2.0 = 0.25
        //   error = 4.0 -> L1: 4.0 - 1.0 = 3.0
        //   Mean = 3.25 / 3 = 1.083333...
        // Row 1: errors [0.0, 0.0, 0.0] -> Mean = 0.0
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().with_beta(2.0).init();

        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0_f32, 1.0, 4.0], [5.0_f32, 5.0, 5.0]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0_f32, 0.0, 0.0], [5.0_f32, 5.0, 5.0]]),
            &device,
        );

        let output = loss.forward_reduce_dims(predictions, targets, &[1]);
        let expected = TensorData::from([[3.25_f32 / 3.0], [0.0]]); // 3.25/3 = 1.0833...
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_smooth_l1_forward_reduce_dims_image_batch() {
        // Simulate per-image Smooth L1 loss for [batch, C, H, W] tensor
        // (common in object detection like Fast R-CNN)
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init(); // beta = 1.0

        // Shape: [2, 1, 2, 2] (batch=2, C=1, H=2, W=2)
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[0.5_f32, 2.0], [0.0, 3.0]]], // Image 1
                [[[1.0_f32, 0.0], [0.5, 1.5]]], // Image 2
            ]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::zeros([2, 1, 2, 2], &device);

        // Reduce over C, H, W (dims 1, 2, 3) to get per-image loss
        let output = loss.forward_reduce_dims(predictions, targets, &[1, 2, 3]);

        // Image 1: losses [[0.125, 1.5], [0.0, 2.5]] -> mean: 4.125 / 4 = 1.03125
        // Image 2: losses [[0.5, 0.0], [0.125, 1.0]] -> mean: 1.625 / 4 = 0.40625
        let expected = TensorData::from([[[[1.03125_f32]]], [[[0.40625_f32]]]]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_smooth_l1_forward_reduce_dims_unsorted() {
        // Test that unsorted dimensions are handled correctly (sorted internally)
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[1.0_f32, 2.0], [3.0, 4.0]], [[5.0_f32, 6.0], [7.0, 8.0]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 3>::zeros([2, 2, 2], &device);

        // Pass dims in reverse order
        let output = loss.forward_reduce_dims(predictions.clone(), targets.clone(), &[2, 1]);
        let expected_output = loss.forward_reduce_dims(predictions, targets, &[1, 2]);

        output
            .into_data()
            .assert_eq(&expected_output.into_data(), false);
    }

    #[test]
    fn test_smooth_l1_forward_reduce_dims_empty_dims() {
        // Reducing over no dimensions should return the unreduced loss
        let device = Default::default();
        let loss = SmoothL1LossConfig::new().init();

        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.5_f32, 2.0], [0.0, 3.0]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2>::zeros([2, 2], &device);

        let loss_reduce_dims = loss.forward_reduce_dims(predictions.clone(), targets.clone(), &[]);
        let loss_no_reduction = loss.forward(predictions, targets);

        loss_reduce_dims
            .into_data()
            .assert_eq(&loss_no_reduction.into_data(), false);
    }
}

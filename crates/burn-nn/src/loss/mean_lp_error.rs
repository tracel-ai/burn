use super::Reduction;
use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};
use burn_core as burn;

/// Configuration for the [Mean Lp Error Loss](MeanLpErrorLoss) module.
///
/// # Example
///
/// ```ignore
/// use burn_nn::loss::{MeanLpErrorLossConfig, Reduction};
///
/// // Create Mean Squared Error (MSE) loss with p=2
/// let mse_loss = MeanLpErrorLossConfig::new(2.0).init();
///
/// // Create Mean Absolute Error (MAE) loss with p=1
/// let mae_loss = MeanLpErrorLossConfig::new(1.0).init();
/// ```
#[derive(Config, Debug)]
pub struct MeanLpErrorLossConfig {
    /// The exponent `p` determining the type of error measurement.
    ///
    /// Common values:
    /// - `p = 1.0`: Mean Absolute Error (MAE) - robust to outliers
    /// - `p = 2.0`: Mean Squared Error (MSE) - standard choice, differentiable everywhere
    /// - `p > 2.0`: Increasingly sensitive to large errors (outliers)
    /// - `0 < p < 1`: More robust to outliers than MAE (quasi-norm)
    pub p: f64,
}

impl MeanLpErrorLossConfig {
    /// Initializes a [Mean Lp Error Loss](MeanLpErrorLoss) module.
    ///
    /// # Panics
    ///
    /// Panics if `p <= 0`.
    pub fn init(&self) -> MeanLpErrorLoss {
        self.assertions();
        MeanLpErrorLoss { p: self.p }
    }

    fn assertions(&self) {
        assert!(self.p > 0.0, "The order of the norm p must be positive.")
    }
}

/// Computes the Mean(L(p) Norm Error)Loss between predictions and targets.
///
/// This loss function computes the element-wise p-th power of absolute errors,
/// then reduces them via mean or sum.
///
/// # Mathematical Definition
///
/// For predictions `ŷ` and targets `y`, the element-wise loss is:
///
/// ```text
/// Lᵢ = |ŷᵢ - yᵢ|ᵖ
/// ```
///
/// With mean reduction (default), the final loss is:
///
/// ```text
/// L = (1/n) × Σᵢ |ŷᵢ - yᵢ|ᵖ
/// ```
///
/// # Notes
///
/// - This implementation computes `|error|^p`, **not** the Lp norm `(Σ|error|^p)^(1/p)`.
/// - The `p = 1` case uses an optimized `abs()` operation.
/// - The `p = 2` case uses an optimized computation `error * error` instead of `powf`.
///
/// # Example
///
/// ```ignore
/// use burn_nn::loss::{MeanLpErrorLossConfig, Reduction};
/// use burn::tensor::Tensor;
///
/// // Create MSE loss
/// let mse = MeanLpErrorLossConfig::new(2.0).init();
///
/// let predictions: Tensor<Backend, 2> = /* model output */;
/// let targets: Tensor<Backend, 2> = /* ground truth */;
///
/// // Compute loss with mean reduction
/// let reduced_loss = mse.forward(predictions, targets, Reduction::Auto);
///
/// // Compute loss with no reduction
/// let unreduced_loss = mse.forward_no_reduction(predictions, targets);
/// ```
#[derive(Module, Clone, Debug)]
pub struct MeanLpErrorLoss {
    /// The order of the norm (e.g., 1 for L1, 2 for L2).
    /// Equivalently, the exponent `p` for computing `|error|^p`.
    pub p: f64,
}

impl MeanLpErrorLoss {
    /// Computes the element-wise loss `|error|^p` with reduction.
    ///
    /// # Arguments
    ///
    /// * `predictions` - The model's predicted values.
    /// * `targets` - The ground truth target values.
    /// * `reduction` - Specifies how to reduce the element-wise losses:
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
    pub fn forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let unreduced_loss = self.forward_no_reduction(predictions, targets);

        match reduction {
            Reduction::Mean | Reduction::Auto => unreduced_loss.mean(),
            Reduction::Sum => unreduced_loss.sum(),
        }
    }

    /// Computes the element-wise loss `|error|^p` without reduction.
    ///
    /// # Arguments
    ///
    /// * `predictions` - The model's predicted values.
    /// * `targets` - The ground truth target values.
    ///
    /// # Returns
    ///
    /// A tensor of the same shape as the inputs, containing `|prediction - target|^p`
    /// for each element.
    ///
    /// # Shapes
    ///
    /// - predictions: `[...dims]` - Any shape
    /// - targets: `[...dims]` - Must match predictions shape
    /// - output: `[...dims]` - Same shape as inputs
    pub fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let error = predictions.sub(targets);

        // Use simplified/optimized expressions for common cases (p = 1, p = 2)
        if self.p == 1.0 {
            // Mean Absolute Error (MAE)
            error.abs()
        } else if self.p == 2.0 {
            // Mean Squared Error (MSE)
            error.clone().mul(error)
        } else {
            error.abs().powf_scalar(self.p)
        }
    }

    /// Computes the element-wise loss `|error|^p` with reduction over specified dimensions.
    ///
    /// Calculates element-wise `|predictions - targets|^p`, then takes the mean
    /// over the specified dimensions. Useful for per-sample or per-channel losses (e.g., when
    /// working with images).
    ///
    /// Dimensions can be provided in any order. They are sorted internally and
    /// reduced from highest to lowest to ensure indices remain valid.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Image tensor: [batch, C, H, W]
    /// let mse_loss = MeanLpErrorLoss::new(2.0);
    ///
    /// // Per-image MSE for PSNR: reduce over C, H, W → [batch, 1, 1, 1]
    /// let mse_per_image = mse_loss.forward_reduce_dims(predictions, targets, &[1, 2, 3]);
    /// ```
    pub fn forward_reduce_dims<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        dims: &[usize],
    ) -> Tensor<B, D> {
        let error = self.forward_no_reduction(predictions, targets);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_mean_lp_error_loss_p1() {
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [3.0, 2.0]]),
            &device,
        );

        let mse = MeanLpErrorLossConfig::new(1.0).init();
        let loss_no_reduction = mse.forward_no_reduction(predictions.clone(), targets.clone());
        let loss_auto = mse.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = mse.forward(predictions, targets, Reduction::Sum);

        let expected = TensorData::from([[1.0, 1.0], [0.0, 2.0]]);
        loss_no_reduction.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([1.0]);
        loss_auto.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([4.0]);
        loss_sum.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mean_lp_error_loss_p2() {
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [3.0, 2.0]]),
            &device,
        );

        let mse = MeanLpErrorLossConfig::new(2.0).init();
        let loss_no_reduction = mse.forward_no_reduction(predictions.clone(), targets.clone());
        let loss_auto = mse.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = mse.forward(predictions, targets, Reduction::Sum);

        let expected = TensorData::from([[1.0, 1.0], [0.0, 4.0]]);
        loss_no_reduction.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([1.5]);
        loss_auto.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([6.0]);
        loss_sum.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mean_lp_error_loss_p_half() {
        // L0.5 quasi-norm: more robust to outliers than L1
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [3.0, 0.0]]),
            &device,
        );

        let loss = MeanLpErrorLossConfig::new(0.5).init();
        let loss_no_reduction = loss.forward_no_reduction(predictions.clone(), targets.clone());
        let loss_auto = loss.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = loss.forward(predictions, targets, Reduction::Sum);

        // |1-2|^0.5 = 1, |2-1|^0.5 = 1, |3-3|^0.5 = 0, |4-0|^0.5 = 2
        let expected = TensorData::from([[1.0, 1.0], [0.0, 2.0]]);
        loss_no_reduction.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([1.0]);
        loss_auto.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([4.0]);
        loss_sum.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mean_lp_error_loss_p3() {
        // L3 norm: more sensitive to outliers than L2
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [3.0, 2.0]]),
            &device,
        );

        let mse = MeanLpErrorLossConfig::new(3.0).init();
        let loss_no_reduction = mse.forward_no_reduction(predictions.clone(), targets.clone());
        let loss_auto = mse.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = mse.forward(predictions, targets, Reduction::Sum);

        // |1-2|^3 = 1, |2-1|^3 = 1, |3-3|^3 = 0, |4-2|^3 = 8
        let expected = TensorData::from([[1.0, 1.0], [0.0, 8.0]]);
        loss_no_reduction.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([2.5]);
        loss_auto.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([10.0]);
        loss_sum.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mean_lp_error_loss_zero_error() {
        // Test when predictions exactly match targets
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let targets = predictions.clone();

        let loss_fn_p1 = MeanLpErrorLossConfig::new(1.0).init();
        let loss_fn_p2 = MeanLpErrorLossConfig::new(2.0).init();

        let loss_p1 = loss_fn_p1.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_p2 = loss_fn_p2.forward(predictions, targets, Reduction::Auto);

        let expected = TensorData::from([0.0]);
        loss_p1.into_data().assert_eq(&expected, false);
        loss_p2.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mean_lp_error_loss_negative_errors() {
        // Test that negative errors are handled correctly (absolute value)
        let device = Default::default();
        let predictions =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);

        let targets =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([3.0, 4.0, 5.0]), &device);

        let loss_fn = MeanLpErrorLossConfig::new(1.0).init();
        let loss_no_reduction = loss_fn.forward_no_reduction(predictions, targets);

        // All errors are negative: 1-3=-2, 2-4=-2, 3-5=-2, but |error| = 2
        let expected = TensorData::from([2.0, 2.0, 2.0]);
        loss_no_reduction.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mean_lp_error_loss_3d_tensor() {
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            &device,
        );

        let targets = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[0.0, 2.0], [3.0, 5.0]], [[4.0, 6.0], [7.0, 10.0]]]),
            &device,
        );

        let loss_fn = MeanLpErrorLossConfig::new(2.0).init();
        let loss = loss_fn.forward(predictions, targets, Reduction::Auto);

        // Errors: 1, 0, 0, -1, 1, 0, 0, -2
        // Squared: 1, 0, 0, 1, 1, 0, 0, 4
        // Mean: 7/8 = 0.875
        let expected = TensorData::from([0.875]);
        loss.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic(expected = "The order of the norm p must be positive.")]
    fn test_mean_lp_error_loss_negative_p_panics() {
        let _ = MeanLpErrorLossConfig::new(-1.0).init();
    }

    #[test]
    #[should_panic(expected = "The order of the norm p must be positive.")]
    fn test_mean_lp_error_loss_zero_p_panics() {
        let _ = MeanLpErrorLossConfig::new(0.0).init();
    }

    #[test]
    fn test_mean_lp_error_loss_fractional_p() {
        // Test p = 1.5
        let device = Default::default();
        let predictions =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([0.0, 4.0]), &device);

        let targets = Tensor::<TestBackend, 1>::from_data(TensorData::from([1.0, 0.0]), &device);

        let loss_fn = MeanLpErrorLossConfig::new(1.5).init();
        let loss_no_reduction = loss_fn.forward_no_reduction(predictions, targets);

        // |0-1|^1.5 = 1, |4-0|^1.5 = 8
        let expected = TensorData::from([1.0, 8.0]);
        loss_no_reduction.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_forward_reduce_dims_single_dim() {
        let device = Default::default();
        // Shape: [2, 3]
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 2.0, 6.0], [1.0, 5.0, 6.0]]),
            &device,
        );
        let loss_fn = MeanLpErrorLossConfig::new(2.0).init();

        // Reduce over dim 1 -> should give [2, 1] shape
        let loss = loss_fn.forward_reduce_dims(predictions, targets, &[1]);

        // Errors row 0: [1, 0, -3] -> squared: [1, 0, 9] -> mean: 10/3
        // Errors row 1: [3, 0, 0] -> squared: [9, 0, 0] -> mean: 3
        let expected = TensorData::from([[10.0 / 3.0], [3.0]]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_forward_reduce_dims_first_dim() {
        let device = Default::default();
        // Shape: [2, 3]
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 2.0, 6.0], [1.0, 5.0, 6.0]]),
            &device,
        );
        let loss_fn = MeanLpErrorLossConfig::new(2.0).init();

        // Reduce over dim 0 -> should give [1, 3] shape
        let loss = loss_fn.forward_reduce_dims(predictions, targets, &[0]);

        // Squared errors: [[1, 0, 9], [9, 0, 0]]
        // Mean over dim 0: [5, 0, 4.5]
        let expected = TensorData::from([[5.0, 0.0, 4.5]]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_forward_reduce_dims_multiple_dims() {
        let device = Default::default();
        // Shape: [2, 2, 2]
        let predictions = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[0.0, 2.0], [3.0, 6.0]], [[4.0, 6.0], [7.0, 10.0]]]),
            &device,
        );
        let loss_fn = MeanLpErrorLossConfig::new(2.0).init();

        // Reduce over dims 1 and 2 -> should give [2, 1, 1] shape
        let loss = loss_fn.forward_reduce_dims(predictions, targets, &[1, 2]);

        // Batch 0 errors: [[1, 0], [0, -2]] -> squared: [[1, 0], [0, 4]] -> mean: 5/4 = 1.25
        // Batch 1 errors: [[1, 0], [0, -2]] -> squared: [[1, 0], [0, 4]] -> mean: 5/4 = 1.25
        let expected = TensorData::from([[[1.25]], [[1.25]]]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_forward_reduce_dims_all_dims() {
        let device = Default::default();
        // Shape: [2, 2]
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [3.0, 2.0]]),
            &device,
        );
        let loss_fn = MeanLpErrorLossConfig::new(2.0).init();

        // Reduce over all dims -> should give [1, 1] shape
        let loss = loss_fn.forward_reduce_dims(predictions, targets, &[0, 1]);

        // Errors: [[-1, 1], [0, 2]] -> squared: [[1, 1], [0, 4]] -> mean: 1.5
        let expected = TensorData::from([[1.5]]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_forward_reduce_dims_image_batch() {
        // Simulate per-image loss for [batch, C, H, W] tensor (common use case for PSNR)
        let device = Default::default();
        // Shape: [2, 1, 2, 2] (batch=2, C=1, H=2, W=2)
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[1.0, 2.0], [3.0, 4.0]]], // Image 1
                [[[5.0, 6.0], [7.0, 8.0]]], // Image 2
            ]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[0.0, 2.0], [3.0, 6.0]]], // Target 1
                [[[5.0, 5.0], [7.0, 7.0]]], // Target 2
            ]),
            &device,
        );
        let loss_fn = MeanLpErrorLossConfig::new(2.0).init();

        // Reduce over C, H, W (dims 1, 2, 3) to get per-image MSE
        let loss = loss_fn.forward_reduce_dims(predictions, targets, &[1, 2, 3]);

        // Image 1 errors: [[1, 0], [0, -2]] -> squared: [[1, 0], [0, 4]] -> mean: 1.25
        // Image 2 errors: [[0, 1], [0, 1]] -> squared: [[0, 1], [0, 1]] -> mean: 0.5
        let expected = TensorData::from([[[[1.25]]], [[[0.5]]]]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_forward_reduce_dims_with_p1() {
        let device = Default::default();
        // Shape: [2, 3]
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 5.0, 3.0], [1.0, 5.0, 9.0]]),
            &device,
        );
        let loss_fn = MeanLpErrorLossConfig::new(1.0).init();

        // Reduce over dim 1 -> should give [2, 1] shape
        let loss = loss_fn.forward_reduce_dims(predictions, targets, &[1]);

        // Abs errors row 0: [1, 3, 0] -> mean: 4/3
        // Abs errors row 1: [3, 0, 3] -> mean: 2
        let expected = TensorData::from([[4.0 / 3.0], [2.0]]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_forward_reduce_dims_empty_dims() {
        // Reducing over no dimensions should return the unreduced loss
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 2.0], [3.0, 6.0]]),
            &device,
        );
        let loss_fn = MeanLpErrorLossConfig::new(2.0).init();
        let loss_reduce_dims =
            loss_fn.forward_reduce_dims(predictions.clone(), targets.clone(), &[]);
        let loss_no_reduction = loss_fn.forward_no_reduction(predictions, targets);

        // Should be equivalent
        loss_reduce_dims
            .into_data()
            .assert_eq(&loss_no_reduction.into_data(), true);
    }

    #[test]
    fn test_forward_reduce_dims_zero_error() {
        let device = Default::default();
        // Shape: [2, 2, 2]
        let predictions = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            &device,
        );
        let targets = predictions.clone();
        let loss_fn = MeanLpErrorLossConfig::new(2.0).init();
        let loss = loss_fn.forward_reduce_dims(predictions, targets, &[1, 2]);

        // All zeros, reduced to shape: [2, 1, 1]
        let expected = TensorData::from([[[0.0]], [[0.0]]]);
        loss.into_data().assert_eq(&expected, false);
    }
}

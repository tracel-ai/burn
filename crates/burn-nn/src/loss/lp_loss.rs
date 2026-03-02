use super::Reduction;
use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};
use burn_core as burn;

/// Configuration for the [Lp Loss](LpLoss) module.
///
/// # Example
///
/// ```ignore
/// use burn_nn::loss::{LpLossConfig, Reduction};
///
/// // Create L1 loss (MAE when using mean reduction)
/// let l1_loss = LpLossConfig::l1();
///
/// // Create L2 loss (MSE when using mean reduction)
/// let l2_loss = LpLossConfig::l2();
///
/// // Create custom Lp loss with p=3
/// let l3_loss = LpLossConfig::new(3.0).init();
/// ```
#[derive(Config, Debug)]
pub struct LpLossConfig {
    /// The exponent `p` determining the type of error measurement.
    ///
    /// Common values:
    /// - `p = 1.0`: L1 loss (MAE with mean reduction) - robust to outliers
    /// - `p = 2.0`: L2 loss (MSE with mean reduction) - standard choice, differentiable everywhere
    /// - `p > 2.0`: Increasingly sensitive to large errors (outliers)
    /// - `0 < p < 1`: More robust to outliers than L1 (quasi-norm)
    pub p: f64,
}

impl LpLossConfig {
    /// Initializes a [Lp Loss](LpLoss) module.
    ///
    /// # Panics
    ///
    /// Panics if `p <= 0`.
    pub fn init(&self) -> LpLoss {
        self.assertions();
        LpLoss { p: self.p }
    }

    /// Creates L1 loss (p=1).
    ///
    /// When used with `Reduction::Mean`, this computes Mean Absolute Error (MAE).
    /// When used with `Reduction::Sum`, this computes Sum of Absolute Errors (SAE).
    pub fn l1() -> LpLoss {
        LpLoss { p: 1.0 }
    }

    /// Creates L2 loss (p=2).
    ///
    /// When used with `Reduction::Mean`, this computes Mean Squared Error (MSE).
    /// When used with `Reduction::Sum`, this computes Sum of Squared Errors (SSE).
    pub fn l2() -> LpLoss {
        LpLoss { p: 2.0 }
    }

    fn assertions(&self) {
        assert!(self.p > 0.0, "The order of the norm p must be positive.")
    }
}

/// Computes the Lp Loss between predictions and targets.
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
/// use burn_nn::loss::{LpLossConfig, Reduction};
/// use burn::tensor::Tensor;
///
/// // Create L2 loss
/// let l2_loss = LpLossConfig::l2();
///
/// let predictions: Tensor<Backend, 2> = /* model output */;
/// let targets: Tensor<Backend, 2> = /* ground truth */;
///
/// // Compute loss with mean reduction (MSE)
/// let mse = l2_loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
///
/// // Compute loss with sum reduction (SSE)
/// let sse = l2_loss.forward(predictions.clone(), targets.clone(), Reduction::Sum);
///
/// // Compute loss with no reduction
/// let unreduced_l2_loss = l2_loss.forward_no_reduction(predictions, targets);
/// ```
#[derive(Module, Clone, Debug)]
pub struct LpLoss {
    /// The order of the norm (e.g., 1 for L1, 2 for L2).
    /// Equivalently, the exponent `p` for computing `|error|^p`.
    pub p: f64,
}

impl LpLoss {
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
            other => panic!("{other:?} reduction is not supported"),
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
            // L1 loss
            error.abs()
        } else if self.p == 2.0 {
            // L2 loss
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
    /// # Arguments
    ///
    /// * `predictions` - The model's predicted values.
    /// * `targets` - The ground truth target values.
    /// * `dims` - Dimensions to reduce over.
    ///
    /// # Returns
    ///
    /// A tensor with the specified dimensions reduced to size 1.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Image tensor: [batch, C, H, W]
    /// let l2_loss = LpLossConfig::l2();
    ///
    /// // Per-image MSE for PSNR: reduce over C, H, W → [batch, 1, 1, 1]
    /// let mse_per_image = l2_loss.forward_reduce_dims(predictions, targets, &[1, 2, 3]);
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

    #[test]
    fn test_lp_loss_l1_constructor() {
        let loss_func_l1 = LpLossConfig::l1();
        let loss_func_p1 = LpLossConfig::new(1.0).init();
        assert_eq!(loss_func_l1.p, 1.0);
        assert_eq!(loss_func_l1.p, loss_func_p1.p);
    }

    #[test]
    fn test_lp_loss_l2_constructor() {
        let loss_func_l2 = LpLossConfig::l2();
        let loss_func_p2 = LpLossConfig::new(2.0).init();
        assert_eq!(loss_func_l2.p, 2.0);
        assert_eq!(loss_func_l2.p, loss_func_p2.p);
    }

    #[test]
    fn test_lp_loss_l1() {
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [3.0, 2.0]]),
            &device,
        );

        let loss_func = LpLossConfig::l1();
        let loss_no_reduction =
            loss_func.forward_no_reduction(predictions.clone(), targets.clone());
        let loss_auto = loss_func.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = loss_func.forward(predictions, targets, Reduction::Sum);

        let expected = TensorData::from([[1.0, 1.0], [0.0, 2.0]]);
        loss_no_reduction.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([1.0]);
        loss_auto.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([4.0]);
        loss_sum.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_lp_loss_l2() {
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [3.0, 2.0]]),
            &device,
        );

        let loss_func = LpLossConfig::l2();
        let loss_no_reduction =
            loss_func.forward_no_reduction(predictions.clone(), targets.clone());
        let loss_auto = loss_func.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = loss_func.forward(predictions, targets, Reduction::Sum);

        let expected = TensorData::from([[1.0, 1.0], [0.0, 4.0]]);
        loss_no_reduction.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([1.5]);
        loss_auto.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([6.0]);
        loss_sum.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_lp_loss_p_half() {
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

        let loss_func = LpLossConfig::new(0.5).init();
        let loss_no_reduction =
            loss_func.forward_no_reduction(predictions.clone(), targets.clone());
        let loss_auto = loss_func.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = loss_func.forward(predictions, targets, Reduction::Sum);

        // |1-2|^0.5 = 1, |2-1|^0.5 = 1, |3-3|^0.5 = 0, |4-0|^0.5 = 2
        let expected = TensorData::from([[1.0, 1.0], [0.0, 2.0]]);
        loss_no_reduction.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([1.0]);
        loss_auto.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([4.0]);
        loss_sum.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_lp_loss_p3() {
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

        let loss_func = LpLossConfig::new(3.0).init();
        let loss_no_reduction =
            loss_func.forward_no_reduction(predictions.clone(), targets.clone());
        let loss_auto = loss_func.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = loss_func.forward(predictions, targets, Reduction::Sum);

        // |1-2|^3 = 1, |2-1|^3 = 1, |3-3|^3 = 0, |4-2|^3 = 8
        let expected = TensorData::from([[1.0, 1.0], [0.0, 8.0]]);
        loss_no_reduction.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([2.5]);
        loss_auto.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([10.0]);
        loss_sum.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_lp_loss_zero_error() {
        // Test when predictions exactly match targets
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let targets = predictions.clone();

        let loss_func_l1 = LpLossConfig::l1();
        let loss_func_l2 = LpLossConfig::l2();

        let l1_loss = loss_func_l1.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let l2_loss = loss_func_l2.forward(predictions, targets, Reduction::Auto);

        let expected = TensorData::from([0.0]);
        l1_loss.into_data().assert_eq(&expected, false);
        l2_loss.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_lp_loss_negative_errors() {
        // Test that negative errors are handled correctly (absolute value)
        let device = Default::default();
        let predictions =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);
        let targets =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([3.0, 4.0, 5.0]), &device);
        let loss_func_l1 = LpLossConfig::l1();
        let loss_func_p1 = LpLossConfig::new(1.0).init();

        let loss_no_reduction_l1 =
            loss_func_l1.forward_no_reduction(predictions.clone(), targets.clone());
        let loss_no_reduction_p1 = loss_func_p1.forward_no_reduction(predictions, targets);

        // All errors are negative: 1-3=-2, 2-4=-2, 3-5=-2, but |error| = 2
        let expected = TensorData::from([2.0, 2.0, 2.0]);
        loss_no_reduction_l1.into_data().assert_eq(&expected, false);
        loss_no_reduction_p1.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_lp_loss_3d_tensor() {
        let device = Default::default();
        let predictions = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[0.0, 2.0], [3.0, 5.0]], [[4.0, 6.0], [7.0, 10.0]]]),
            &device,
        );
        let loss_func_l2 = LpLossConfig::l2();
        let loss_func_p2 = LpLossConfig::new(2.0).init();

        let loss_l2 = loss_func_l2.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_p2 = loss_func_p2.forward(predictions, targets, Reduction::Auto);

        // Errors: 1, 0, 0, -1, 1, 0, 0, -2
        // Squared: 1, 0, 0, 1, 1, 0, 0, 4
        // Mean: 7/8 = 0.875
        let expected = TensorData::from([0.875]);
        loss_l2.into_data().assert_eq(&expected, false);
        loss_p2.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic(expected = "The order of the norm p must be positive.")]
    fn test_lp_loss_negative_p_panics() {
        let _ = LpLossConfig::new(-1.0).init();
    }

    #[test]
    #[should_panic(expected = "The order of the norm p must be positive.")]
    fn test_lp_loss_zero_p_panics() {
        let _ = LpLossConfig::new(0.0).init();
    }

    #[test]
    fn test_lp_loss_fractional_p() {
        // Test p = 1.5
        let device = Default::default();
        let predictions =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([0.0, 4.0]), &device);

        let targets = Tensor::<TestBackend, 1>::from_data(TensorData::from([1.0, 0.0]), &device);

        let loss_func = LpLossConfig::new(1.5).init();
        let loss_no_reduction = loss_func.forward_no_reduction(predictions, targets);

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
        let loss_func_l2 = LpLossConfig::l2();
        let loss_func_p2 = LpLossConfig::new(2.0).init();

        // Reduce over dim 1 -> should give [2, 1] shape
        let loss_l2 = loss_func_l2.forward_reduce_dims(predictions.clone(), targets.clone(), &[1]);
        let loss_p2 = loss_func_p2.forward_reduce_dims(predictions, targets, &[1]);

        // Errors row 0: [1, 0, -3] -> squared: [1, 0, 9] -> mean: 10/3
        // Errors row 1: [3, 0, 0] -> squared: [9, 0, 0] -> mean: 3
        let expected = TensorData::from([[10.0 / 3.0], [3.0]]);
        loss_l2
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
        loss_p2
            .into_data()
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
        let loss_func = LpLossConfig::l2();

        // Reduce over dim 0 -> should give [1, 3] shape
        let loss = loss_func.forward_reduce_dims(predictions, targets, &[0]);

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
        let loss_func = LpLossConfig::l2();

        // Reduce over dims 1 and 2 -> should give [2, 1, 1] shape
        let loss = loss_func.forward_reduce_dims(predictions, targets, &[1, 2]);

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
        let loss_func = LpLossConfig::l2();

        // Reduce over all dims -> should give [1, 1] shape
        let loss = loss_func.forward_reduce_dims(predictions, targets, &[0, 1]);

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
        let loss_func = LpLossConfig::l2();

        // Reduce over C, H, W (dims 1, 2, 3) to get per-image MSE
        let loss = loss_func.forward_reduce_dims(predictions, targets, &[1, 2, 3]);

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
        let loss_func = LpLossConfig::l1();

        // Reduce over dim 1 -> should give [2, 1] shape
        let loss = loss_func.forward_reduce_dims(predictions, targets, &[1]);

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
        let loss_func = LpLossConfig::l2();
        let loss_reduce_dims =
            loss_func.forward_reduce_dims(predictions.clone(), targets.clone(), &[]);
        let loss_no_reduction = loss_func.forward_no_reduction(predictions, targets);

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
        let loss_func = LpLossConfig::l2();
        let loss = loss_func.forward_reduce_dims(predictions, targets, &[1, 2]);

        // All zeros, reduced to shape: [2, 1, 1]
        let expected = TensorData::from([[[0.0]], [[0.0]]]);
        loss.into_data().assert_eq(&expected, false);
    }
}

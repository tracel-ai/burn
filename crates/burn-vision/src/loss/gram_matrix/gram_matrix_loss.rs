use burn_core as burn;

use super::vgg19::Vgg19;
use super::weights::load_vgg19_weights;
use crate::loss::Reduction;
use burn::config::Config;
use burn::module::Module;
use burn::tensor::{Device, Tensor};

/// Configuration for the [Gram Matrix Loss](GramMatrixLoss) module.
///
/// Gram Matrix Loss (often used in Neural Style Transfer) measures the difference in
/// texture or style between two images. It does this by comparing the spatial correlations
/// of their feature maps extracted from a pretrained VGG19 network.
///
/// # Example
///
/// ```rust,ignore
/// use burn_nn::loss::pretrained::gram_matrix::GramMatrixLossConfig;
///
/// // Create Gram Matrix Loss with equal weights for all 5 layers
/// let device = Default::default();
/// let gram_loss = GramMatrixLossConfig::new(vec![1.0, 1.0, 1.0, 1.0, 1.0])
///     .with_use_avg_pool(true)
///     .init::<B>(&device);
/// ```
///
/// # Reference
/// [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
#[cfg_attr(docsrs, doc(cfg(feature = "pretrained")))]
#[derive(Config, Debug)]
pub struct GramMatrixLossConfig {
    /// The weights of the layer contributing to the total loss.
    /// Should have a length of 5 since Gram Matrix Loss uses 5 specific VGG19 layers.
    pub layer_weights: Vec<f32>,

    /// If true, uses average pooling in the VGG19 feature extractor.
    /// If false, uses the max pooling.
    #[config(default = "false")]
    pub use_avg_pool: bool,
}

impl GramMatrixLossConfig {
    /// Initializes a [Gram Matrix Loss](GramMatrixLoss) module.
    ///
    /// This will automatically download and load the pretrained VGG19 weights
    /// if they are not already cached locally.
    ///
    /// # Panics
    ///
    /// - If `layer_weights` does not contain exactly 5 elements.
    /// - If any of the weights in `layer_weights` is not non-negative.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use burn_nn::loss::pretrained::gram_matrix::GramMatrixLossConfig;
    ///
    /// // Create Gram Matrix Loss with equal weights for all 5 layers
    /// let device = Default::default();
    /// let gram_loss = GramMatrixLossConfig::new(vec![1.0, 1.0, 1.0, 1.0, 1.0])
    ///     .init::<B>(&device);
    /// ```
    pub fn init(&self, device: &Device) -> GramMatrixLoss<B> {
        self.assertions();

        let vgg19 = Vgg19::new(self.use_avg_pool, device);
        let pretrained_vgg19 = load_vgg19_weights(vgg19).no_grad();

        GramMatrixLoss {
            layer_weights: self.layer_weights.clone(),
            feat_extractor: pretrained_vgg19,
        }
    }

    fn assertions(&self) {
        assert!(
            self.layer_weights.len() == 5,
            "The layer_weights vector must contain exactly 5 elements"
        );
        assert!(
            self.layer_weights.iter().all(|&w| w >= 0.0),
            "All layer weights must be non-negative"
        );
    }
}

/// Computes the Gram Matrix Loss between predictions and targets.
///
/// This loss function extracts features from 5 specific layers of a pretrained VGG19 network
/// (`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`). It computes the Gram matrix for each
/// layer's feature map, which captures the style/texture information, and calculates the
/// Mean Squared Error between the Gram matrices of the predictions and targets.
///
/// # Note
///
/// The Gram Matrix Loss assumes the input tensors are already in the \[0.0, 1.0\] range.
///
/// # Example
///
/// ```rust,ignore
/// use burn_nn::loss::pretrained::gram_matrix::GramMatrixLossConfig;
///
/// // Initialize the loss function via its config
/// let device = Default::default();
/// // Uses max pool by default
/// let loss_fn = GramMatrixLossConfig::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]).init::<B>(&device);
/// ```
///
/// # Reference
/// [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
#[cfg_attr(docsrs, doc(cfg(feature = "pretrained")))]
#[derive(Module, Debug)]
pub struct GramMatrixLoss {
    /// The weights of the layer contributing to the total loss.
    /// Should have a length of 5 since Gram Matrix Loss uses 5 layers.
    pub layer_weights: Vec<f32>,
    /// Pretrained VGG19 feature extractor
    pub feat_extractor: Vgg19<B>,
}

impl GramMatrixLoss<B> {
    /// Computes the Gram Matrix Loss with reduction.
    ///
    /// # Arguments
    ///
    /// - `predictions` - The model's predicted images. The pixels should be in the \[0.0, 1.0\] range.
    /// - `targets` - The ground truth target images. The pixels should be in the \[0.0, 1.0\] range.
    /// - `reduction` - Specifies how to reduce the batch losses.
    ///   - `Reduction::Mean` or `Reduction::Auto`: Returns the mean of batch losses.
    ///   - `Reduction::Sum`: Returns the sum of batch losses.
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the reduced loss value.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, 3, height, width]`
    /// - targets: `[batch_size, 3, height, width]`
    /// - output: `[1]`
    ///
    /// # Panics
    ///
    /// - If the `reduction` type is not supported.
    /// - If the input tensors do not have exactly 3 channels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use burn_nn::loss::pretrained::gram_matrix::GramMatrixLossConfig;
    /// use burn::loss::Reduction;
    ///
    /// let device = Default::default();
    /// let loss_fn = GramMatrixLossConfig::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]).init::<B>(&device);
    ///
    /// let predictions = /* [N, 3, H, W] */;
    /// let targets = /* [N, 3, H, W] */;
    ///
    /// # Returns a tensor with shape [1] containing a single loss value
    /// let loss = loss_fn.forward(predictions, targets, Reduction::Mean);
    /// ```
    pub fn forward(
        &self,
        predictions: Tensor<4>,
        targets: Tensor<4>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let unreduced_loss = self.forward_no_reduction(predictions, targets);

        match reduction {
            Reduction::Mean | Reduction::Auto => unreduced_loss.mean(),
            Reduction::Sum => unreduced_loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Computes the unreduced Gram Matrix Loss per sample in the batch.
    ///
    /// # Arguments
    ///
    /// - `predictions` - The model's predicted images. The pixels should be in the \[0.0, 1.0\] range.
    /// - `targets` - The ground truth target images. The pixels should be in the \[0.0, 1.0\] range.
    ///
    /// # Returns
    ///
    /// A 1D tensor containing the total weighted loss for each sample in the batch.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, 3, height, width]`
    /// - targets: `[batch_size, 3, height, width]`
    /// - output: `[batch_size]`
    ///
    /// # Panics
    ///
    /// - If the input tensors do not have exactly 3 channels.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use burn_nn::loss::pretrained::gram_matrix::GramMatrixLossConfig;
    ///
    /// let device = Default::default();
    /// let loss_fn = GramMatrixLossConfig::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]).init::<B>(&device);
    ///
    /// let predictions = /* [N, 3, H, W] */;
    /// let targets = /* [N, 3, H, W] */;
    ///
    /// // Returns a tensor of shape [N] containing the loss for each sample
    /// let unreduced_loss = loss_fn.forward_no_reduction(predictions, targets);
    /// ```
    pub fn forward_no_reduction(&self, predictions: Tensor<4>, targets: Tensor<4>) -> Tensor<1> {
        let pred_processed = self.preprocess_input(predictions);
        let target_processed = self.preprocess_input(targets);

        // Both vectors contain 5 entries since there are 5 layers
        // Both feature map tensors already have the shape [N, C, H * W]
        let pred_features = self.feat_extractor.forward(pred_processed);
        let mut pred_normalization_factors = Vec::with_capacity(5);
        for feature_tensor in &pred_features {
            let [_, c, h_times_w] = feature_tensor.dims();
            let (c_f, hw_f) = (c as f32, h_times_w as f32);
            pred_normalization_factors.push(4.0 * c_f * c_f * hw_f * hw_f);
        }

        let target_features = self.feat_extractor.forward(target_processed);

        // Create vector which will hold loss tensors for each layer
        let mut loss_tensors = Vec::with_capacity(pred_features.len());

        // Compute and add the weighted loss for each layer to the final loss tensor.
        // Note that the loss tensor for each layer and the final loss tensors
        // contains a loss value for each sample in the batch.
        for (pred_f, target_f) in pred_features.into_iter().zip(target_features) {
            // Compute Gram matrix as G = F(F^T)
            // [N, C, H*W] times [N, H*W, C] equals [N, C, C]
            let pred_gram_matrices = pred_f.clone().matmul(pred_f.clone().transpose());
            let target_gram_matrices = target_f.clone().matmul(target_f.clone().transpose());

            let gram_matrices_diff = pred_gram_matrices - target_gram_matrices;
            let gram_matrices_diff_squared = gram_matrices_diff.powi_scalar(2);

            // For each sample, sum over all the entries of the gram matrix.
            // Equivalently, sum over the last two dimensions (the two C dimensions).
            let loss = gram_matrices_diff_squared
                .sum_dims(&[1, 2])
                .squeeze_dims::<1>(&[1, 2]);
            loss_tensors.push(loss);
        }

        // Sum each layer's loss in the vector of loss tensors
        let scaled_loss_tensors: Vec<Tensor<1>> = loss_tensors
            .into_iter()
            .zip(pred_normalization_factors)
            .zip(self.layer_weights.clone())
            .map(|((loss_tensor, norm_factor), weight)| {
                loss_tensor.div_scalar(norm_factor).mul_scalar(weight)
            })
            .collect();
        let stacked_loss_tensors = Tensor::stack::<2>(scaled_loss_tensors, 1);
        stacked_loss_tensors.sum_dim(1).squeeze_dim(1)
    }

    /// Applies standard ImageNet normalization to the input tensor for the VGG19 network.
    ///
    /// # Note
    ///
    /// This method assumes the input tensor is already in the \[0.0, 1.0\] range.
    ///
    /// # Panics
    ///
    /// - If the input tensor does not have exactly 3 channels.
    fn preprocess_input(&self, tensor: Tensor<4>) -> Tensor<4> {
        let device = &tensor.device();
        let channels = tensor.dims()[1];
        assert!(
            channels == 3,
            "Expected input tensor to have exactly 3 channels, but got {}",
            channels
        );

        // ImageNet normalization constants
        let mean = Tensor::<1>::from_floats([0.485, 0.456, 0.406], device).reshape([1, 3, 1, 1]);
        let std = Tensor::<1>::from_floats([0.229, 0.224, 0.225], device).reshape([1, 3, 1, 1]);

        (tensor - mean) / std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    #[should_panic(expected = "The layer_weights vector must contain exactly 5 elements")]
    fn test_gram_matrix_loss_config_invalid_length() {
        let device = Default::default();
        GramMatrixLossConfig::new(vec![1.0, 1.0]).init(&device);
    }

    #[test]
    #[should_panic(expected = "All layer weights must be non-negative")]
    fn test_gram_matrix_loss_config_negative_weights() {
        let device = Default::default();
        GramMatrixLossConfig::new(vec![1.0, -1.0, 1.0, 1.0, 1.0]).init(&device);
    }

    #[test]
    // TODO: run tests only locally, and #[serial]'ised?
    // #[cfg(feature = "test-local")]
    #[ignore = "downloads pre-trained weights"]
    fn test_gram_matrix_loss_config_valid_weights() {
        let device = Default::default();
        let layer_weights = vec![0.0, 0.2, 0.2, 0.25, 0.4];
        let loss_fn = GramMatrixLossConfig::new(layer_weights.clone()).init(&device);
        assert_eq!(
            loss_fn.layer_weights, layer_weights,
            "Expected layer weights vector {:?}, got {:?}",
            loss_fn.layer_weights, layer_weights
        );
    }

    #[test]
    #[should_panic(expected = "Expected input tensor to have exactly 3 channels, but got 1")]
    fn test_gram_matrix_loss_1_channel_panic() {
        let device = Default::default();
        let loss_fn = GramMatrixLoss {
            layer_weights: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            feat_extractor: Vgg19::new(false, &device),
        };

        // 1 channel (Grayscale) should panic
        let tensor1: Tensor<4> = Tensor::random([2, 1, 16, 16], Distribution::Default, &device);
        let tensor2 = tensor1.clone();

        let _ = loss_fn.forward(tensor1, tensor2, Reduction::Mean);
    }

    #[test]
    #[should_panic(expected = "Expected input tensor to have exactly 3 channels, but got 4")]
    fn test_gram_matrix_loss_4_channel_panic() {
        let device = Default::default();
        let loss_fn = GramMatrixLoss {
            layer_weights: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            feat_extractor: Vgg19::new(false, &device),
        };

        // 4 channels (e.g., RGBA) should panic
        let tensor1: Tensor<4> = Tensor::random([2, 4, 16, 16], Distribution::Default, &device);
        let tensor2 = tensor1.clone();

        let _ = loss_fn.forward(tensor1, tensor2, Reduction::Mean);
    }

    #[test]
    fn test_gram_matrix_loss_zero_for_identical_inputs() {
        let device = Default::default();

        // Instantiate using Vgg19::new() to use random weights
        let loss_fn = GramMatrixLoss {
            layer_weights: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            feat_extractor: Vgg19::new(false, &device),
        };

        let tensor1: Tensor<4> = Tensor::random([2, 3, 16, 16], Distribution::Default, &device);
        let tensor2 = tensor1.clone();

        let loss = loss_fn.forward(tensor1, tensor2, Reduction::Mean);
        let loss_val: f32 = loss.into_scalar();

        // Loss should be exactly 0 (or extremely close due to floating point) when inputs are identical
        assert!(
            loss_val.abs() < 1e-4,
            "Loss should be zero for identical inputs"
        );
    }

    #[test]
    fn test_gram_matrix_loss_greater_than_zero_for_different_inputs() {
        let device = Default::default();
        let loss_fn = GramMatrixLoss {
            layer_weights: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            feat_extractor: Vgg19::new(false, &device),
        };

        let tensor1: Tensor<4> = Tensor::ones([2, 3, 16, 16], &device);
        let tensor2: Tensor<4> = Tensor::zeros([2, 3, 16, 16], &device);

        let loss = loss_fn.forward(tensor1, tensor2, Reduction::Mean);
        let loss_val: f32 = loss.into_scalar();

        assert!(
            loss_val > 0.0,
            "Loss should be positive for different inputs"
        );
    }

    #[test]
    fn test_gram_matrix_loss_forward_no_reduction_shape() {
        let device = Default::default();
        let loss_fn = GramMatrixLoss {
            layer_weights: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            feat_extractor: Vgg19::new(false, &device),
        };

        let batch_size = 4;
        let tensor1: Tensor<4> =
            Tensor::random([batch_size, 3, 16, 16], Distribution::Default, &device);
        let tensor2: Tensor<4> =
            Tensor::random([batch_size, 3, 16, 16], Distribution::Default, &device);

        let unreduced_loss = loss_fn.forward_no_reduction(tensor1, tensor2);

        // Unreduced loss should return a 1D tensor with shape [batch_size]
        assert_eq!(unreduced_loss.dims(), [batch_size]);
    }

    #[test]
    fn test_gram_matrix_loss_reduction_sum_vs_mean() {
        let device = Default::default();
        let loss_fn = GramMatrixLoss {
            layer_weights: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            feat_extractor: Vgg19::new(false, &device),
        };

        let batch_size = 4;
        let tensor1: Tensor<4> =
            Tensor::random([batch_size, 3, 16, 16], Distribution::Default, &device);
        let tensor2: Tensor<4> =
            Tensor::random([batch_size, 3, 16, 16], Distribution::Default, &device);

        let loss_mean: f32 = loss_fn
            .forward(tensor1.clone(), tensor2.clone(), Reduction::Mean)
            .into_scalar();
        let loss_sum: f32 = loss_fn
            .forward(tensor1, tensor2, Reduction::Sum)
            .into_scalar();

        let expected_sum = loss_mean * (batch_size as f32);
        let diff = (loss_sum - expected_sum).abs();

        // The sum reduction should be equal to the mean reduction multiplied by the batch size
        assert!(
            diff < 1e-4,
            "Sum reduction should equal batch_size * Mean reduction"
        );
    }

    #[test]
    fn test_gram_matrix_loss_with_avg_pool() {
        let device = Default::default();
        let loss_fn = GramMatrixLoss {
            layer_weights: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            // Initialize with use_avg_pool = true
            feat_extractor: Vgg19::new(true, &device),
        };

        let batch_size = 4;
        let tensor1: Tensor<4> = Tensor::ones([batch_size, 3, 16, 16], &device);
        let tensor2: Tensor<4> = Tensor::zeros([batch_size, 3, 16, 16], &device);

        let loss = loss_fn.forward(tensor1, tensor2, Reduction::Mean);
        let loss_val: f32 = loss.into_scalar();

        assert!(
            loss_val > 0.0,
            "Loss should be positive for different inputs using avg pooling"
        );
    }

    #[test]
    fn test_gram_matrix_loss_autodiff() {
        let device = Device::default().autodiff();
        let loss_fn = GramMatrixLoss {
            layer_weights: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            feat_extractor: Vgg19::new(false, &device).no_grad(),
        };

        // The prediction tensor requires gradients
        let predictions: Tensor<4> = Tensor::ones([2, 3, 16, 16], &device).require_grad();

        // The target tensor does not require gradients
        let targets: Tensor<4> = Tensor::zeros([2, 3, 16, 16], &device);

        let loss = loss_fn.forward(predictions.clone(), targets, Reduction::Mean);
        let grads = loss.backward();

        // Verify that gradients were successfully computed for the predictions tensor
        let pred_grad = predictions.grad(&grads);
        assert!(
            pred_grad.is_some(),
            "Gradients should be computed for the predictions tensor"
        );

        // Verify that VGG19 parameters do not have gradients
        let conv1_1_weight_grad = loss_fn.feat_extractor.conv1_1.weight.val().grad(&grads);
        assert!(
            conv1_1_weight_grad.is_none(),
            "Gradients should not be computed for VGG19 parameters"
        );
    }

    #[test]
    #[ignore = "downloads pre-trained weights"]
    fn test_gram_matrix_loss_pretrained_weights_identical_inputs() {
        let device = Default::default();
        let loss_fn = GramMatrixLossConfig::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]).init(&device);

        let tensor1: Tensor<4> = Tensor::random([2, 3, 16, 16], Distribution::Default, &device);
        let tensor2 = tensor1.clone();

        let loss = loss_fn.forward(tensor1, tensor2, Reduction::Mean);
        let loss_val: f32 = loss.into_scalar();

        // Loss should be exactly 0 (or extremely close due to floating point) when inputs are identical
        assert!(
            loss_val.abs() < 1e-4,
            "Loss should be zero for identical inputs"
        );
    }

    #[test]
    #[ignore = "downloads pre-trained weights"]
    fn test_gram_matrix_loss_pretrained_weights_different_inputs() {
        let device = Default::default();
        let loss_fn = GramMatrixLossConfig::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]).init(&device);

        let tensor1: Tensor<4> = Tensor::ones([2, 3, 16, 16], &device);
        let tensor2: Tensor<4> = Tensor::zeros([2, 3, 16, 16], &device);

        let loss = loss_fn.forward(tensor1, tensor2, Reduction::Mean);
        let loss_val: f32 = loss.into_scalar();

        assert!(
            loss_val > 0.0,
            "Loss should be positive for different inputs"
        );
    }
}

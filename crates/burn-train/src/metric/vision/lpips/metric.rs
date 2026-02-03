//! LPIPS (Learned Perceptual Image Patch Similarity) metric module.
//!
//! LPIPS measures perceptual similarity between images using deep features.
//! Supports VGG16, AlexNet, and SqueezeNet as backbone networks.
//!
//! Reference: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
//! <https://arxiv.org/abs/1801.03924>

use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_nn::conv::{Conv2d, Conv2dConfig};
use burn_nn::loss::Reduction;

use super::alexnet::AlexFeatureExtractor;
use super::squeezenet::SqueezeFeatureExtractor;
use super::vgg::VggFeatureExtractor;

/// Network type for LPIPS.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum LpipsNet {
    /// VGG16 network (default)
    Vgg,
    /// AlexNet network
    Alex,
    /// SqueezeNet network
    Squeeze,
}

/// Configuration for [Lpips](Lpips) metric module.
///
/// # Example
///
/// ```ignore
/// use burn_train::metric::vision::{LpipsConfig, LpipsNet};
///
/// // VGG (default)
/// let lpips_vgg = LpipsConfig::new().init(&device);
///
/// // AlexNet
/// let lpips_alex = LpipsConfig::new()
///     .with_net(LpipsNet::Alex)
///     .init(&device);
///
/// // SqueezeNet
/// let lpips_squeeze = LpipsConfig::new()
///     .with_net(LpipsNet::Squeeze)
///     .init(&device);
/// ```
#[derive(Config, Debug)]
pub struct LpipsConfig {
    /// Network type for feature extraction.
    #[config(default = "LpipsNet::Vgg")]
    pub net: LpipsNet,

    /// Whether to normalize input images to [-1, 1] range.
    /// Set to true if input is in [0, 1] range.
    #[config(default = true)]
    pub normalize: bool,
}

impl LpipsConfig {
    /// Initialize a new [Lpips](Lpips) module.
    ///
    /// # Arguments
    ///
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A new LPIPS module. Weights should be loaded from pretrained model for accurate results.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Lpips<B> {
        match self.net {
            LpipsNet::Vgg => {
                // Channel sizes for VGG16: [64, 128, 256, 512, 512]
                Lpips::Vgg(LpipsVgg {
                    extractor: VggFeatureExtractor::new(device),
                    lin0: Conv2dConfig::new([64, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin1: Conv2dConfig::new([128, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin2: Conv2dConfig::new([256, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin3: Conv2dConfig::new([512, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin4: Conv2dConfig::new([512, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    normalize: self.normalize,
                })
            }
            LpipsNet::Alex => {
                // Channel sizes for AlexNet: [64, 192, 384, 256, 256]
                Lpips::Alex(LpipsAlex {
                    extractor: AlexFeatureExtractor::new(device),
                    lin0: Conv2dConfig::new([64, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin1: Conv2dConfig::new([192, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin2: Conv2dConfig::new([384, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin3: Conv2dConfig::new([256, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin4: Conv2dConfig::new([256, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    normalize: self.normalize,
                })
            }
            LpipsNet::Squeeze => {
                // Channel sizes for SqueezeNet: [64, 128, 256, 384, 384, 512, 512]
                Lpips::Squeeze(LpipsSqueeze {
                    extractor: SqueezeFeatureExtractor::new(device),
                    lin0: Conv2dConfig::new([64, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin1: Conv2dConfig::new([128, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin2: Conv2dConfig::new([256, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin3: Conv2dConfig::new([384, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin4: Conv2dConfig::new([384, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin5: Conv2dConfig::new([512, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    lin6: Conv2dConfig::new([512, 1], [1, 1])
                        .with_bias(false)
                        .init(device),
                    normalize: self.normalize,
                })
            }
        }
    }
}

/// LPIPS (Learned Perceptual Image Patch Similarity) metric module.
///
/// Computes perceptual distance between two images using deep features.
/// Supports VGG16, AlexNet, and SqueezeNet as backbone networks.
///
/// # Example
///
/// ```ignore
/// use burn_train::metric::vision::{LpipsConfig, LpipsNet, Reduction};
///
/// let device = Default::default();
/// let lpips = LpipsConfig::new().init(&device);
///
/// let img1: Tensor<B, 4> = /* [batch, 3, H, W] */;
/// let img2: Tensor<B, 4> = /* [batch, 3, H, W] */;
///
/// // Compute LPIPS distance
/// let distance = lpips.forward(img1, img2, Reduction::Mean);
/// ```
#[derive(Module, Debug)]
#[module(custom_display)]
pub enum Lpips<B: Backend> {
    /// VGG16 backbone (5 feature layers)
    Vgg(LpipsVgg<B>),
    /// AlexNet backbone (5 feature layers)
    Alex(LpipsAlex<B>),
    /// SqueezeNet backbone (7 feature layers)
    Squeeze(LpipsSqueeze<B>),
}

/// LPIPS with VGG16 backbone.
#[derive(Module, Debug)]
pub struct LpipsVgg<B: Backend> {
    /// VGG feature extractor
    pub(crate) extractor: VggFeatureExtractor<B>,
    /// Linear layers for each feature level
    pub(crate) lin0: Conv2d<B>,
    pub(crate) lin1: Conv2d<B>,
    pub(crate) lin2: Conv2d<B>,
    pub(crate) lin3: Conv2d<B>,
    pub(crate) lin4: Conv2d<B>,
    /// Whether to normalize input
    pub(crate) normalize: bool,
}

/// LPIPS with AlexNet backbone.
#[derive(Module, Debug)]
pub struct LpipsAlex<B: Backend> {
    /// AlexNet feature extractor
    pub(crate) extractor: AlexFeatureExtractor<B>,
    /// Linear layers for each feature level
    pub(crate) lin0: Conv2d<B>,
    pub(crate) lin1: Conv2d<B>,
    pub(crate) lin2: Conv2d<B>,
    pub(crate) lin3: Conv2d<B>,
    pub(crate) lin4: Conv2d<B>,
    /// Whether to normalize input
    pub(crate) normalize: bool,
}

/// LPIPS with SqueezeNet backbone.
#[derive(Module, Debug)]
pub struct LpipsSqueeze<B: Backend> {
    /// SqueezeNet feature extractor
    pub(crate) extractor: SqueezeFeatureExtractor<B>,
    /// Linear layers for each feature level
    pub(crate) lin0: Conv2d<B>,
    pub(crate) lin1: Conv2d<B>,
    pub(crate) lin2: Conv2d<B>,
    pub(crate) lin3: Conv2d<B>,
    pub(crate) lin4: Conv2d<B>,
    pub(crate) lin5: Conv2d<B>,
    pub(crate) lin6: Conv2d<B>,
    /// Whether to normalize input
    pub(crate) normalize: bool,
}

impl<B: Backend> LpipsVgg<B> {
    /// Compute LPIPS distance without reduction using VGG backbone.
    pub fn forward_no_reduction(&self, input: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let [batch, _, _, _] = input.dims();

        // Preprocess inputs
        let (input, target) = preprocess_inputs(input, target, self.normalize);

        // Extract features from both images
        let feats0 = self.extractor.forward(input);
        let feats1 = self.extractor.forward(target);

        // Compute distance for each layer
        let device = feats0[0].device();
        let mut total_loss = Tensor::zeros([batch], &device);

        total_loss = total_loss.add(compute_layer_distance(&feats0[0], &feats1[0], &self.lin0));
        total_loss = total_loss.add(compute_layer_distance(&feats0[1], &feats1[1], &self.lin1));
        total_loss = total_loss.add(compute_layer_distance(&feats0[2], &feats1[2], &self.lin2));
        total_loss = total_loss.add(compute_layer_distance(&feats0[3], &feats1[3], &self.lin3));
        total_loss = total_loss.add(compute_layer_distance(&feats0[4], &feats1[4], &self.lin4));

        total_loss
    }
}

impl<B: Backend> LpipsAlex<B> {
    /// Compute LPIPS distance without reduction using AlexNet backbone.
    pub fn forward_no_reduction(&self, input: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let [batch, _, _, _] = input.dims();

        // Preprocess inputs
        let (input, target) = preprocess_inputs(input, target, self.normalize);

        // Extract features from both images
        let feats0 = self.extractor.forward(input);
        let feats1 = self.extractor.forward(target);

        // Compute distance for each layer
        let device = feats0[0].device();
        let mut total_loss = Tensor::zeros([batch], &device);

        total_loss = total_loss.add(compute_layer_distance(&feats0[0], &feats1[0], &self.lin0));
        total_loss = total_loss.add(compute_layer_distance(&feats0[1], &feats1[1], &self.lin1));
        total_loss = total_loss.add(compute_layer_distance(&feats0[2], &feats1[2], &self.lin2));
        total_loss = total_loss.add(compute_layer_distance(&feats0[3], &feats1[3], &self.lin3));
        total_loss = total_loss.add(compute_layer_distance(&feats0[4], &feats1[4], &self.lin4));

        total_loss
    }
}

impl<B: Backend> LpipsSqueeze<B> {
    /// Compute LPIPS distance without reduction using SqueezeNet backbone.
    pub fn forward_no_reduction(&self, input: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let [batch, _, _, _] = input.dims();

        // Preprocess inputs
        let (input, target) = preprocess_inputs(input, target, self.normalize);

        // Extract features from both images
        let feats0 = self.extractor.forward(input);
        let feats1 = self.extractor.forward(target);

        // Compute distance for each layer (7 layers for SqueezeNet)
        let device = feats0[0].device();
        let mut total_loss = Tensor::zeros([batch], &device);

        total_loss = total_loss.add(compute_layer_distance(&feats0[0], &feats1[0], &self.lin0));
        total_loss = total_loss.add(compute_layer_distance(&feats0[1], &feats1[1], &self.lin1));
        total_loss = total_loss.add(compute_layer_distance(&feats0[2], &feats1[2], &self.lin2));
        total_loss = total_loss.add(compute_layer_distance(&feats0[3], &feats1[3], &self.lin3));
        total_loss = total_loss.add(compute_layer_distance(&feats0[4], &feats1[4], &self.lin4));
        total_loss = total_loss.add(compute_layer_distance(&feats0[5], &feats1[5], &self.lin5));
        total_loss = total_loss.add(compute_layer_distance(&feats0[6], &feats1[6], &self.lin6));

        total_loss
    }
}

impl<B: Backend> ModuleDisplay for Lpips<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let (net_name, normalize) = match self {
            Lpips::Vgg(inner) => ("Vgg", inner.normalize),
            Lpips::Alex(inner) => ("Alex", inner.normalize),
            Lpips::Squeeze(inner) => ("Squeeze", inner.normalize),
        };
        content
            .add("net", &format!("{}", net_name))
            .add("normalize", &format!("{}", normalize))
            .optional()
    }
}

impl<B: Backend> Lpips<B> {
    /// Compute LPIPS distance with reduction.
    ///
    /// # Arguments
    ///
    /// * `input` - First image tensor of shape `[batch, 3, H, W]`
    /// * `target` - Second image tensor of shape `[batch, 3, H, W]`
    /// * `reduction` - How to reduce the output (Mean, Sum, or Auto)
    ///
    /// # Returns
    ///
    /// Scalar tensor of shape `[1]`.
    ///
    /// # Shapes
    ///
    /// - input: `[batch, 3, H, W]`
    /// - target: `[batch, 3, H, W]`
    /// - output: `[1]`
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        target: Tensor<B, 4>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let distance = self.forward_no_reduction(input, target);

        match reduction {
            Reduction::Mean | Reduction::Auto | Reduction::BatchMean => distance.mean(),
            Reduction::Sum => distance.sum(),
        }
    }

    /// Compute LPIPS distance without reduction.
    ///
    /// # Arguments
    ///
    /// * `input` - First image tensor of shape `[batch, 3, H, W]`
    /// * `target` - Second image tensor of shape `[batch, 3, H, W]`
    ///
    /// # Returns
    ///
    /// Per-sample distance tensor of shape `[batch]`.
    ///
    /// # Shapes
    ///
    /// - input: `[batch, 3, H, W]`
    /// - target: `[batch, 3, H, W]`
    /// - output: `[batch]`
    pub fn forward_no_reduction(&self, input: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        match self {
            Lpips::Vgg(inner) => inner.forward_no_reduction(input, target),
            Lpips::Alex(inner) => inner.forward_no_reduction(input, target),
            Lpips::Squeeze(inner) => inner.forward_no_reduction(input, target),
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Normalize tensor to unit norm along channel dimension.
fn normalize_tensor<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let norm = x.clone().mul(x.clone()).sum_dim(1).sqrt().clamp_min(1e-10);
    x.div(norm)
}

/// Apply ImageNet normalization used by PyTorch lpips.
/// shift = [-.030, -.088, -.188], scale = [.458, .448, .450]
/// output = (input - shift) / scale
fn scaling_layer<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let device = x.device();
    let [batch, _, h, w] = x.dims();

    // Create shift and scale tensors [1, 3, 1, 1] and broadcast
    let shift = Tensor::<B, 2>::from_floats([[-0.030], [-0.088], [-0.188]], &device)
        .reshape([1, 3, 1, 1])
        .expand([batch, 3, h, w]);
    let scale = Tensor::<B, 2>::from_floats([[0.458], [0.448], [0.450]], &device)
        .reshape([1, 3, 1, 1])
        .expand([batch, 3, h, w]);

    x.sub(shift).div(scale)
}

/// Compute normalized L2 distance for a single layer.
fn compute_layer_distance<B: Backend>(
    feat0: &Tensor<B, 4>,
    feat1: &Tensor<B, 4>,
    lin: &Conv2d<B>,
) -> Tensor<B, 1> {
    // Normalize features (unit norm along channel dimension)
    let feat0_norm = normalize_tensor(feat0.clone());
    let feat1_norm = normalize_tensor(feat1.clone());

    // Compute squared difference
    let diff = feat0_norm.sub(feat1_norm);
    let diff_sq = diff.clone().mul(diff);

    // Apply linear layer (learned weights)
    // Shape: [batch, C, H, W] -> [batch, 1, H, W]
    // Note: With pretrained weights, lin weights are positive.
    // We use abs() to ensure non-negative output with random weights.
    let weighted = lin.forward(diff_sq).abs();

    // Spatial average: compute mean over C, H, W dimensions
    // Shape: [batch, 1, H, W] -> [batch]
    let [batch, c, h, w] = weighted.dims();

    // Reshape to [batch, c*h*w] then take mean over last dimension
    weighted
        .reshape([batch, c * h * w])
        .mean_dim(1)
        .squeeze_dim::<1>(1)
}

/// Preprocess input images for LPIPS computation.
fn preprocess_inputs<B: Backend>(
    input: Tensor<B, 4>,
    target: Tensor<B, 4>,
    normalize: bool,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // Normalize to [-1, 1] if needed
    let (input, target) = if normalize {
        (
            input.mul_scalar(2.0).sub_scalar(1.0),
            target.mul_scalar(2.0).sub_scalar(1.0),
        )
    } else {
        (input, target)
    };

    // Apply ImageNet normalization (same as PyTorch lpips scaling_layer)
    (scaling_layer(input), scaling_layer(target))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::{ops::FloatElem, TensorData, Tolerance};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;
    type FT = FloatElem<TestBackend>;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    // =========================================================================
    // Basic Functionality Tests
    // =========================================================================

    /// Identical images should have LPIPS distance of 0.
    #[test]
    fn test_lpips_identical_images_zero_distance() {
        let device = Default::default();
        let image = TestTensor::<4>::ones([1, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let distance = lpips.forward(image.clone(), image, Reduction::Mean);

        // Identical images → distance = 0
        let expected = TensorData::from([0.0]);
        distance
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    /// Different images should have LPIPS distance greater than 0.
    #[test]
    fn test_lpips_different_images_positive_distance() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let distance = lpips.forward(image1, image2, Reduction::Mean);

        let distance_value = distance.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            distance_value > 0.0,
            "LPIPS should be > 0 for different images"
        );
    }

    /// Test symmetry: LPIPS(a, b) == LPIPS(b, a).
    #[test]
    fn test_lpips_symmetry() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let distance_ab = lpips.forward(image1.clone(), image2.clone(), Reduction::Mean);
        let distance_ba = lpips.forward(image2, image1, Reduction::Mean);

        distance_ab
            .into_data()
            .assert_approx_eq::<FT>(&distance_ba.into_data(), Tolerance::default());
    }

    // =========================================================================
    // Reduction Tests
    // =========================================================================

    #[test]
    fn test_lpips_forward_mean_reduction() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([2, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([2, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let distance = lpips.forward(image1, image2, Reduction::Mean);

        assert_eq!(distance.dims(), [1]);
    }

    #[test]
    fn test_lpips_forward_no_reduction() {
        let device = Default::default();

        let batch_size = 4;
        let image1 = TestTensor::<4>::zeros([batch_size, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([batch_size, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let distance = lpips.forward_no_reduction(image1, image2);

        assert_eq!(distance.dims(), [batch_size]);
    }

    // =========================================================================
    // Output Range Tests
    // =========================================================================

    #[test]
    fn test_lpips_output_non_negative() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let distance = lpips.forward(image1, image2, Reduction::Mean);

        let distance_value = distance.into_data().to_vec::<f32>().unwrap()[0];
        assert!(distance_value >= 0.0, "LPIPS should be >= 0");
    }

    // =========================================================================
    // AlexNet Tests
    // =========================================================================

    /// Test AlexNet LPIPS with identical images.
    #[test]
    fn test_lpips_alex_identical_images_zero_distance() {
        let device = Default::default();
        let image = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Alex)
            .init(&device);
        let distance = lpips.forward(image.clone(), image, Reduction::Mean);

        let expected = TensorData::from([0.0]);
        distance
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    /// Test AlexNet LPIPS with different images.
    #[test]
    fn test_lpips_alex_different_images_positive_distance() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 64, 64], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Alex)
            .init(&device);
        let distance = lpips.forward(image1, image2, Reduction::Mean);

        let distance_value = distance.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            distance_value > 0.0,
            "LPIPS (Alex) should be > 0 for different images"
        );
    }

    // =========================================================================
    // SqueezeNet Tests
    // =========================================================================

    /// Test SqueezeNet LPIPS with identical images.
    #[test]
    fn test_lpips_squeeze_identical_images_zero_distance() {
        let device = Default::default();
        let image = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Squeeze)
            .init(&device);
        let distance = lpips.forward(image.clone(), image, Reduction::Mean);

        let expected = TensorData::from([0.0]);
        distance
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    /// Test SqueezeNet LPIPS with different images.
    #[test]
    fn test_lpips_squeeze_different_images_positive_distance() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 64, 64], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Squeeze)
            .init(&device);
        let distance = lpips.forward(image1, image2, Reduction::Mean);

        let distance_value = distance.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            distance_value > 0.0,
            "LPIPS (Squeeze) should be > 0 for different images"
        );
    }

    // =========================================================================
    // Display Tests
    // =========================================================================

    #[test]
    fn display_vgg() {
        let device = Default::default();
        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);

        let display_str = format!("{lpips}");
        assert!(display_str.contains("Lpips"));
        assert!(display_str.contains("Vgg"));
    }

    #[test]
    fn display_alex() {
        let device = Default::default();
        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Alex)
            .init(&device);

        let display_str = format!("{lpips}");
        assert!(display_str.contains("Lpips"));
        assert!(display_str.contains("Alex"));
    }

    #[test]
    fn display_squeeze() {
        let device = Default::default();
        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Squeeze)
            .init(&device);

        let display_str = format!("{lpips}");
        assert!(display_str.contains("Lpips"));
        assert!(display_str.contains("Squeeze"));
    }
}

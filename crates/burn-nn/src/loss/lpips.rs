//! LPIPS (Learned Perceptual Image Patch Similarity) loss module.
//!
//! LPIPS measures perceptual similarity between images using deep features.
//! Supports VGG16, AlexNet, and SqueezeNet as backbone networks.
//!
//! Reference: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
//! <https://arxiv.org/abs/1801.03924>
//!
//! # Loading Pretrained Weights
//!
//! To use LPIPS with pretrained weights from PyTorch:
//!
//! 1. Save weights in Python:
//! ```python
//! import torch, lpips
//! # For VGG
//! loss_fn = lpips.LPIPS(net='vgg')
//! torch.save(loss_fn.state_dict(), 'lpips_vgg.pt')
//! # For AlexNet
//! loss_fn = lpips.LPIPS(net='alex')
//! torch.save(loss_fn.state_dict(), 'lpips_alex.pt')
//! # For SqueezeNet
//! loss_fn = lpips.LPIPS(net='squeeze')
//! torch.save(loss_fn.state_dict(), 'lpips_squeeze.pt')
//! ```
//!
//! 2. Load weights in Rust (see [`lpips_key_remaps`] for key remapping)

use alloc::vec::Vec;

use burn_core as burn;

use super::Reduction;
use crate::conv::{Conv2d, Conv2dConfig};
use crate::PaddingConfig2d;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

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

/// Configuration for [Lpips](Lpips) loss module.
///
/// # Example
///
/// ```ignore
/// use burn_nn::loss::{LpipsConfig, LpipsNet};
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
                let vgg = VggFeatureExtractor::new(device);
                // Channel sizes for VGG16: [64, 128, 256, 512, 512]
                Lpips {
                    vgg: Some(vgg),
                    alex: None,
                    squeeze: None,
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
                    lin5: None,
                    lin6: None,
                    normalize: self.normalize,
                    net: Ignored(self.net),
                }
            }
            LpipsNet::Alex => {
                let alex = AlexFeatureExtractor::new(device);
                // Channel sizes for AlexNet: [64, 192, 384, 256, 256]
                Lpips {
                    vgg: None,
                    alex: Some(alex),
                    squeeze: None,
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
                    lin5: None,
                    lin6: None,
                    normalize: self.normalize,
                    net: Ignored(self.net),
                }
            }
            LpipsNet::Squeeze => {
                let squeeze = SqueezeFeatureExtractor::new(device);
                // Channel sizes for SqueezeNet: [64, 128, 256, 384, 384, 512, 512]
                Lpips {
                    vgg: None,
                    alex: None,
                    squeeze: Some(squeeze),
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
                    lin5: Some(
                        Conv2dConfig::new([512, 1], [1, 1])
                            .with_bias(false)
                            .init(device),
                    ),
                    lin6: Some(
                        Conv2dConfig::new([512, 1], [1, 1])
                            .with_bias(false)
                            .init(device),
                    ),
                    normalize: self.normalize,
                    net: Ignored(self.net),
                }
            }
        }
    }
}

/// LPIPS (Learned Perceptual Image Patch Similarity) loss module.
///
/// Computes perceptual distance between two images using deep features.
/// Supports VGG16, AlexNet, and SqueezeNet as backbone networks.
///
/// # Example
///
/// ```ignore
/// use burn_nn::loss::{LpipsConfig, LpipsNet, Reduction};
///
/// let device = Default::default();
/// let lpips = LpipsConfig::new().init(&device);
///
/// let img1: Tensor<B, 4> = /* [batch, 3, H, W] */;
/// let img2: Tensor<B, 4> = /* [batch, 3, H, W] */;
///
/// // Compute LPIPS distance
/// let loss = lpips.forward(img1, img2, Reduction::Mean);
/// ```
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Lpips<B: Backend> {
    /// VGG feature extractor (used when net=Vgg)
    vgg: Option<VggFeatureExtractor<B>>,
    /// AlexNet feature extractor (used when net=Alex)
    alex: Option<AlexFeatureExtractor<B>>,
    /// SqueezeNet feature extractor (used when net=Squeeze)
    squeeze: Option<SqueezeFeatureExtractor<B>>,
    /// Linear layer for layer 0 features
    lin0: Conv2d<B>,
    /// Linear layer for layer 1 features
    lin1: Conv2d<B>,
    /// Linear layer for layer 2 features
    lin2: Conv2d<B>,
    /// Linear layer for layer 3 features
    lin3: Conv2d<B>,
    /// Linear layer for layer 4 features
    lin4: Conv2d<B>,
    /// Linear layer for layer 5 features (SqueezeNet only)
    lin5: Option<Conv2d<B>>,
    /// Linear layer for layer 6 features (SqueezeNet only)
    lin6: Option<Conv2d<B>>,
    /// Whether to normalize input
    normalize: bool,
    /// Network type
    net: Ignored<LpipsNet>,
}

impl<B: Backend> ModuleDisplay for Lpips<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("net", &alloc::format!("{:?}", self.net.0))
            .add("normalize", &self.normalize)
            .optional()
    }
}

/// Key remapping rules for loading PyTorch lpips VGG weights.
const LPIPS_VGG_KEY_REMAPS: &[(&str, &str)] = &[
    // VGG Block 1: net.slice1
    ("net\\.slice1\\.0\\.(.*)", "vgg.conv1_1.$1"),
    ("net\\.slice1\\.2\\.(.*)", "vgg.conv1_2.$1"),
    // VGG Block 2: net.slice2
    ("net\\.slice2\\.5\\.(.*)", "vgg.conv2_1.$1"),
    ("net\\.slice2\\.7\\.(.*)", "vgg.conv2_2.$1"),
    // VGG Block 3: net.slice3
    ("net\\.slice3\\.10\\.(.*)", "vgg.conv3_1.$1"),
    ("net\\.slice3\\.12\\.(.*)", "vgg.conv3_2.$1"),
    ("net\\.slice3\\.14\\.(.*)", "vgg.conv3_3.$1"),
    // VGG Block 4: net.slice4
    ("net\\.slice4\\.17\\.(.*)", "vgg.conv4_1.$1"),
    ("net\\.slice4\\.19\\.(.*)", "vgg.conv4_2.$1"),
    ("net\\.slice4\\.21\\.(.*)", "vgg.conv4_3.$1"),
    // VGG Block 5: net.slice5
    ("net\\.slice5\\.24\\.(.*)", "vgg.conv5_1.$1"),
    ("net\\.slice5\\.26\\.(.*)", "vgg.conv5_2.$1"),
    ("net\\.slice5\\.28\\.(.*)", "vgg.conv5_3.$1"),
    // Linear layers: lin0.model.1 -> lin0
    ("lin0\\.model\\.1\\.(.*)", "lin0.$1"),
    ("lin1\\.model\\.1\\.(.*)", "lin1.$1"),
    ("lin2\\.model\\.1\\.(.*)", "lin2.$1"),
    ("lin3\\.model\\.1\\.(.*)", "lin3.$1"),
    ("lin4\\.model\\.1\\.(.*)", "lin4.$1"),
];

/// Key remapping rules for loading PyTorch lpips AlexNet weights.
const LPIPS_ALEX_KEY_REMAPS: &[(&str, &str)] = &[
    // AlexNet layers: slice1.0 -> conv1, slice2.3 -> conv2, etc.
    // The indices (0, 3, 6, 8, 10) are positions in the original AlexNet features Sequential
    ("net\\.slice1\\.0\\.(.*)", "alex.conv1.$1"),
    ("net\\.slice2\\.3\\.(.*)", "alex.conv2.$1"),
    ("net\\.slice3\\.6\\.(.*)", "alex.conv3.$1"),
    ("net\\.slice4\\.8\\.(.*)", "alex.conv4.$1"),
    ("net\\.slice5\\.10\\.(.*)", "alex.conv5.$1"),
    // Linear layers: lin0.model.1 -> lin0
    ("lin0\\.model\\.1\\.(.*)", "lin0.$1"),
    ("lin1\\.model\\.1\\.(.*)", "lin1.$1"),
    ("lin2\\.model\\.1\\.(.*)", "lin2.$1"),
    ("lin3\\.model\\.1\\.(.*)", "lin3.$1"),
    ("lin4\\.model\\.1\\.(.*)", "lin4.$1"),
];

/// Key remapping rules for loading PyTorch lpips SqueezeNet weights.
const LPIPS_SQUEEZE_KEY_REMAPS: &[(&str, &str)] = &[
    // SqueezeNet conv1: slice1.0
    ("net\\.slice1\\.0\\.(.*)", "squeeze.conv1.$1"),
    // Fire modules: squeeze.3, 4, 6, 7, 9, 10, 11, 12
    // slice2: features 2-4 (MaxPool, Fire3, Fire4) -> fire1, fire2
    ("net\\.slice2\\.3\\.squeeze\\.(.*)", "squeeze.fire1.squeeze.$1"),
    ("net\\.slice2\\.3\\.expand1x1\\.(.*)", "squeeze.fire1.expand1x1.$1"),
    ("net\\.slice2\\.3\\.expand3x3\\.(.*)", "squeeze.fire1.expand3x3.$1"),
    ("net\\.slice2\\.4\\.squeeze\\.(.*)", "squeeze.fire2.squeeze.$1"),
    ("net\\.slice2\\.4\\.expand1x1\\.(.*)", "squeeze.fire2.expand1x1.$1"),
    ("net\\.slice2\\.4\\.expand3x3\\.(.*)", "squeeze.fire2.expand3x3.$1"),
    // slice3: features 5-7 (MaxPool, Fire5, Fire6) -> fire3, fire4
    ("net\\.slice3\\.6\\.squeeze\\.(.*)", "squeeze.fire3.squeeze.$1"),
    ("net\\.slice3\\.6\\.expand1x1\\.(.*)", "squeeze.fire3.expand1x1.$1"),
    ("net\\.slice3\\.6\\.expand3x3\\.(.*)", "squeeze.fire3.expand3x3.$1"),
    ("net\\.slice3\\.7\\.squeeze\\.(.*)", "squeeze.fire4.squeeze.$1"),
    ("net\\.slice3\\.7\\.expand1x1\\.(.*)", "squeeze.fire4.expand1x1.$1"),
    ("net\\.slice3\\.7\\.expand3x3\\.(.*)", "squeeze.fire4.expand3x3.$1"),
    // slice4: features 8-9 (MaxPool, Fire7) -> fire5
    ("net\\.slice4\\.9\\.squeeze\\.(.*)", "squeeze.fire5.squeeze.$1"),
    ("net\\.slice4\\.9\\.expand1x1\\.(.*)", "squeeze.fire5.expand1x1.$1"),
    ("net\\.slice4\\.9\\.expand3x3\\.(.*)", "squeeze.fire5.expand3x3.$1"),
    // slice5: features 10 (Fire8) -> fire6
    ("net\\.slice5\\.10\\.squeeze\\.(.*)", "squeeze.fire6.squeeze.$1"),
    ("net\\.slice5\\.10\\.expand1x1\\.(.*)", "squeeze.fire6.expand1x1.$1"),
    ("net\\.slice5\\.10\\.expand3x3\\.(.*)", "squeeze.fire6.expand3x3.$1"),
    // slice6: features 11 (Fire9) -> fire7
    ("net\\.slice6\\.11\\.squeeze\\.(.*)", "squeeze.fire7.squeeze.$1"),
    ("net\\.slice6\\.11\\.expand1x1\\.(.*)", "squeeze.fire7.expand1x1.$1"),
    ("net\\.slice6\\.11\\.expand3x3\\.(.*)", "squeeze.fire7.expand3x3.$1"),
    // slice7: features 12 (Fire10) -> fire8
    ("net\\.slice7\\.12\\.squeeze\\.(.*)", "squeeze.fire8.squeeze.$1"),
    ("net\\.slice7\\.12\\.expand1x1\\.(.*)", "squeeze.fire8.expand1x1.$1"),
    ("net\\.slice7\\.12\\.expand3x3\\.(.*)", "squeeze.fire8.expand3x3.$1"),
    // Linear layers: lin0-lin6.model.1 -> lin0-lin6
    ("lin0\\.model\\.1\\.(.*)", "lin0.$1"),
    ("lin1\\.model\\.1\\.(.*)", "lin1.$1"),
    ("lin2\\.model\\.1\\.(.*)", "lin2.$1"),
    ("lin3\\.model\\.1\\.(.*)", "lin3.$1"),
    ("lin4\\.model\\.1\\.(.*)", "lin4.$1"),
    ("lin5\\.model\\.1\\.(.*)", "lin5.$1"),
    ("lin6\\.model\\.1\\.(.*)", "lin6.$1"),
];

/// Get key remapping rules for loading PyTorch lpips weights.
///
/// These patterns map PyTorch lpips library keys to burn LPIPS module keys.
/// Use with `burn_import::pytorch::LoadArgs::with_key_remap()`.
///
/// # Arguments
///
/// * `net` - The network type to get remapping rules for.
///
/// # Returns
///
/// A slice of (pattern, replacement) tuples for key remapping.
///
/// # Example
///
/// ```ignore
/// use burn::record::{FullPrecisionSettings, Recorder};
/// use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
/// use burn_nn::loss::{LpipsConfig, LpipsNet, lpips_key_remaps};
///
/// // For VGG:
/// // >>> loss_fn = lpips.LPIPS(net='vgg')
/// // >>> torch.save(loss_fn.state_dict(), 'lpips_vgg.pt')
///
/// // For AlexNet:
/// // >>> loss_fn = lpips.LPIPS(net='alex')
/// // >>> torch.save(loss_fn.state_dict(), 'lpips_alex.pt')
///
/// let device = Default::default();
/// let net = LpipsNet::Alex;
///
/// let mut load_args = LoadArgs::new("lpips_alex.pt".into());
/// for (pattern, replacement) in lpips_key_remaps(net) {
///     load_args = load_args.with_key_remap(pattern, replacement);
/// }
///
/// let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
///     .load(load_args, &device)
///     .expect("Failed to load weights");
///
/// let lpips = LpipsConfig::new()
///     .with_net(net)
///     .init::<Backend>(&device)
///     .load_record(record);
/// ```
pub fn lpips_key_remaps(net: LpipsNet) -> &'static [(&'static str, &'static str)] {
    match net {
        LpipsNet::Vgg => LPIPS_VGG_KEY_REMAPS,
        LpipsNet::Alex => LPIPS_ALEX_KEY_REMAPS,
        LpipsNet::Squeeze => LPIPS_SQUEEZE_KEY_REMAPS,
    }
}

impl<B: Backend> Lpips<B> {
    /// Compute LPIPS loss with reduction.
    ///
    /// # Arguments
    ///
    /// * `input` - First image tensor of shape `[batch, 3, H, W]`
    /// * `target` - Second image tensor of shape `[batch, 3, H, W]`
    /// * `reduction` - How to reduce the loss (Mean, Sum, or Auto)
    ///
    /// # Returns
    ///
    /// Scalar loss tensor of shape `[1]`.
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
        let loss = self.forward_no_reduction(input, target);

        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }

    /// Compute LPIPS loss without reduction.
    ///
    /// # Arguments
    ///
    /// * `input` - First image tensor of shape `[batch, 3, H, W]`
    /// * `target` - Second image tensor of shape `[batch, 3, H, W]`
    ///
    /// # Returns
    ///
    /// Per-sample loss tensor of shape `[batch]`.
    ///
    /// # Shapes
    ///
    /// - input: `[batch, 3, H, W]`
    /// - target: `[batch, 3, H, W]`
    /// - output: `[batch]`
    pub fn forward_no_reduction(&self, input: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let [batch, _, _, _] = input.dims();

        // Normalize to [-1, 1] if needed
        let (input, target) = if self.normalize {
            (
                input.mul_scalar(2.0).sub_scalar(1.0),
                target.mul_scalar(2.0).sub_scalar(1.0),
            )
        } else {
            (input, target)
        };

        // Apply ImageNet normalization (same as PyTorch lpips scaling_layer)
        // shift = [-.030, -.088, -.188], scale = [.458, .448, .450]
        // output = (input - shift) / scale
        let input = Self::scaling_layer(input);
        let target = Self::scaling_layer(target);

        // Extract features from both images using the appropriate network
        let (feats0, feats1) = match self.net.0 {
            LpipsNet::Vgg => {
                let vgg = self.vgg.as_ref().expect("VGG extractor not initialized");
                (vgg.forward(input), vgg.forward(target))
            }
            LpipsNet::Alex => {
                let alex = self.alex.as_ref().expect("Alex extractor not initialized");
                (alex.forward(input), alex.forward(target))
            }
            LpipsNet::Squeeze => {
                let squeeze = self
                    .squeeze
                    .as_ref()
                    .expect("Squeeze extractor not initialized");
                (squeeze.forward(input), squeeze.forward(target))
            }
        };

        // Compute distance for each layer
        let device = feats0[0].device();
        let mut total_loss = Tensor::zeros([batch], &device);

        // Layer 0
        let diff0 = self.compute_layer_distance(&feats0[0], &feats1[0], &self.lin0);
        total_loss = total_loss.add(diff0);

        // Layer 1
        let diff1 = self.compute_layer_distance(&feats0[1], &feats1[1], &self.lin1);
        total_loss = total_loss.add(diff1);

        // Layer 2
        let diff2 = self.compute_layer_distance(&feats0[2], &feats1[2], &self.lin2);
        total_loss = total_loss.add(diff2);

        // Layer 3
        let diff3 = self.compute_layer_distance(&feats0[3], &feats1[3], &self.lin3);
        total_loss = total_loss.add(diff3);

        // Layer 4
        let diff4 = self.compute_layer_distance(&feats0[4], &feats1[4], &self.lin4);
        total_loss = total_loss.add(diff4);

        // Layers 5-6 (SqueezeNet only)
        if let (Some(lin5), Some(lin6)) = (&self.lin5, &self.lin6) {
            let diff5 = self.compute_layer_distance(&feats0[5], &feats1[5], lin5);
            total_loss = total_loss.add(diff5);

            let diff6 = self.compute_layer_distance(&feats0[6], &feats1[6], lin6);
            total_loss = total_loss.add(diff6);
        }

        total_loss
    }

    /// Compute normalized L2 distance for a single layer.
    fn compute_layer_distance(
        &self,
        feat0: &Tensor<B, 4>,
        feat1: &Tensor<B, 4>,
        lin: &Conv2d<B>,
    ) -> Tensor<B, 1> {
        // Normalize features (unit norm along channel dimension)
        let feat0_norm = Self::normalize_tensor(feat0.clone());
        let feat1_norm = Self::normalize_tensor(feat1.clone());

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

    /// Normalize tensor to unit norm along channel dimension.
    fn normalize_tensor(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let norm = x.clone().mul(x.clone()).sum_dim(1).sqrt().clamp_min(1e-10);
        x.div(norm)
    }

    /// Apply ImageNet normalization used by PyTorch lpips.
    /// shift = [-.030, -.088, -.188], scale = [.458, .448, .450]
    /// output = (input - shift) / scale
    fn scaling_layer(x: Tensor<B, 4>) -> Tensor<B, 4> {
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
}

// =============================================================================
// VGG16 Feature Extractor
// =============================================================================

/// VGG16 feature extractor for LPIPS.
///
/// Extracts features from 5 layers:
/// - conv1_2: 64 channels
/// - conv2_2: 128 channels
/// - conv3_3: 256 channels
/// - conv4_3: 512 channels
/// - conv5_3: 512 channels
#[derive(Module, Debug)]
pub struct VggFeatureExtractor<B: Backend> {
    // Block 1
    conv1_1: Conv2d<B>,
    conv1_2: Conv2d<B>,
    // Block 2
    conv2_1: Conv2d<B>,
    conv2_2: Conv2d<B>,
    // Block 3
    conv3_1: Conv2d<B>,
    conv3_2: Conv2d<B>,
    conv3_3: Conv2d<B>,
    // Block 4
    conv4_1: Conv2d<B>,
    conv4_2: Conv2d<B>,
    conv4_3: Conv2d<B>,
    // Block 5
    conv5_1: Conv2d<B>,
    conv5_2: Conv2d<B>,
    conv5_3: Conv2d<B>,
}

impl<B: Backend> VggFeatureExtractor<B> {
    /// Create a new VGG16 feature extractor.
    pub fn new(device: &B::Device) -> Self {
        let conv_config = |in_ch, out_ch| {
            Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .with_bias(true)
        };

        Self {
            // Block 1: 3 -> 64
            conv1_1: conv_config(3, 64).init(device),
            conv1_2: conv_config(64, 64).init(device),
            // Block 2: 64 -> 128
            conv2_1: conv_config(64, 128).init(device),
            conv2_2: conv_config(128, 128).init(device),
            // Block 3: 128 -> 256
            conv3_1: conv_config(128, 256).init(device),
            conv3_2: conv_config(256, 256).init(device),
            conv3_3: conv_config(256, 256).init(device),
            // Block 4: 256 -> 512
            conv4_1: conv_config(256, 512).init(device),
            conv4_2: conv_config(512, 512).init(device),
            conv4_3: conv_config(512, 512).init(device),
            // Block 5: 512 -> 512
            conv5_1: conv_config(512, 512).init(device),
            conv5_2: conv_config(512, 512).init(device),
            conv5_3: conv_config(512, 512).init(device),
        }
    }

    /// Extract features from 5 VGG layers.
    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut features = Vec::with_capacity(5);

        // Block 1
        let x = relu(self.conv1_1.forward(x));
        let x = relu(self.conv1_2.forward(x));
        features.push(x.clone());
        let x = max_pool2d(x);

        // Block 2
        let x = relu(self.conv2_1.forward(x));
        let x = relu(self.conv2_2.forward(x));
        features.push(x.clone());
        let x = max_pool2d(x);

        // Block 3
        let x = relu(self.conv3_1.forward(x));
        let x = relu(self.conv3_2.forward(x));
        let x = relu(self.conv3_3.forward(x));
        features.push(x.clone());
        let x = max_pool2d(x);

        // Block 4
        let x = relu(self.conv4_1.forward(x));
        let x = relu(self.conv4_2.forward(x));
        let x = relu(self.conv4_3.forward(x));
        features.push(x.clone());
        let x = max_pool2d(x);

        // Block 5
        let x = relu(self.conv5_1.forward(x));
        let x = relu(self.conv5_2.forward(x));
        let x = relu(self.conv5_3.forward(x));
        features.push(x);

        features
    }
}

// =============================================================================
// AlexNet Feature Extractor
// =============================================================================

/// AlexNet feature extractor for LPIPS.
///
/// Extracts features from 5 layers:
/// - conv1: 64 channels (after ReLU)
/// - conv2: 192 channels (after ReLU)
/// - conv3: 384 channels (after ReLU)
/// - conv4: 256 channels (after ReLU)
/// - conv5: 256 channels (after ReLU)
#[derive(Module, Debug)]
pub struct AlexFeatureExtractor<B: Backend> {
    /// Conv1: 3 -> 64, kernel 11x11, stride 4, padding 2
    conv1: Conv2d<B>,
    /// Conv2: 64 -> 192, kernel 5x5, stride 1, padding 2
    conv2: Conv2d<B>,
    /// Conv3: 192 -> 384, kernel 3x3, stride 1, padding 1
    conv3: Conv2d<B>,
    /// Conv4: 384 -> 256, kernel 3x3, stride 1, padding 1
    conv4: Conv2d<B>,
    /// Conv5: 256 -> 256, kernel 3x3, stride 1, padding 1
    conv5: Conv2d<B>,
}

impl<B: Backend> AlexFeatureExtractor<B> {
    /// Create a new AlexNet feature extractor.
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Conv1: 3 -> 64, 11x11, stride 4, padding 2
            conv1: Conv2dConfig::new([3, 64], [11, 11])
                .with_stride([4, 4])
                .with_padding(PaddingConfig2d::Explicit(2, 2))
                .with_bias(true)
                .init(device),
            // Conv2: 64 -> 192, 5x5, stride 1, padding 2
            conv2: Conv2dConfig::new([64, 192], [5, 5])
                .with_padding(PaddingConfig2d::Explicit(2, 2))
                .with_bias(true)
                .init(device),
            // Conv3: 192 -> 384, 3x3, stride 1, padding 1
            conv3: Conv2dConfig::new([192, 384], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .init(device),
            // Conv4: 384 -> 256, 3x3, stride 1, padding 1
            conv4: Conv2dConfig::new([384, 256], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .init(device),
            // Conv5: 256 -> 256, 3x3, stride 1, padding 1
            conv5: Conv2dConfig::new([256, 256], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .init(device),
        }
    }

    /// Extract features from 5 AlexNet layers.
    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut features = Vec::with_capacity(5);

        // Slice 1: Conv1 + ReLU
        let x = relu(self.conv1.forward(x));
        features.push(x.clone());

        // Slice 2: MaxPool + Conv2 + ReLU
        let x = max_pool2d_alex(x);
        let x = relu(self.conv2.forward(x));
        features.push(x.clone());

        // Slice 3: MaxPool + Conv3 + ReLU
        let x = max_pool2d_alex(x);
        let x = relu(self.conv3.forward(x));
        features.push(x.clone());

        // Slice 4: Conv4 + ReLU (no pooling)
        let x = relu(self.conv4.forward(x));
        features.push(x.clone());

        // Slice 5: Conv5 + ReLU (no pooling)
        let x = relu(self.conv5.forward(x));
        features.push(x);

        features
    }
}

// =============================================================================
// SqueezeNet Feature Extractor
// =============================================================================

/// Fire module for SqueezeNet.
///
/// A fire module consists of:
/// - Squeeze layer: 1x1 conv to reduce channels
/// - Expand layers: parallel 1x1 and 3x3 convs, concatenated
#[derive(Module, Debug)]
pub struct FireModule<B: Backend> {
    /// Squeeze layer: 1x1 conv
    squeeze: Conv2d<B>,
    /// Expand 1x1 conv
    expand1x1: Conv2d<B>,
    /// Expand 3x3 conv
    expand3x3: Conv2d<B>,
}

impl<B: Backend> FireModule<B> {
    /// Create a new Fire module.
    pub fn new(
        in_channels: usize,
        squeeze_channels: usize,
        expand1x1_channels: usize,
        expand3x3_channels: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            squeeze: Conv2dConfig::new([in_channels, squeeze_channels], [1, 1])
                .with_bias(true)
                .init(device),
            expand1x1: Conv2dConfig::new([squeeze_channels, expand1x1_channels], [1, 1])
                .with_bias(true)
                .init(device),
            expand3x3: Conv2dConfig::new([squeeze_channels, expand3x3_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .init(device),
        }
    }

    /// Forward pass through fire module.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let squeezed = relu(self.squeeze.forward(x));
        let e1 = relu(self.expand1x1.forward(squeezed.clone()));
        let e3 = relu(self.expand3x3.forward(squeezed));
        // Concatenate along channel dimension
        Tensor::cat(alloc::vec![e1, e3], 1)
    }
}

/// SqueezeNet 1.1 feature extractor for LPIPS.
///
/// Extracts features from 7 layers:
/// - After conv1+relu: 64 channels
/// - After fire1+fire2: 128 channels
/// - After fire3+fire4: 256 channels
/// - After fire5: 384 channels
/// - After fire6: 384 channels
/// - After fire7: 512 channels
/// - After fire8: 512 channels
#[derive(Module, Debug)]
pub struct SqueezeFeatureExtractor<B: Backend> {
    /// Conv1: 3 -> 64, kernel 3x3, stride 2
    conv1: Conv2d<B>,
    /// Fire1: 64 -> 128 (squeeze=16, expand=64+64)
    fire1: FireModule<B>,
    /// Fire2: 128 -> 128 (squeeze=16, expand=64+64)
    fire2: FireModule<B>,
    /// Fire3: 128 -> 256 (squeeze=32, expand=128+128)
    fire3: FireModule<B>,
    /// Fire4: 256 -> 256 (squeeze=32, expand=128+128)
    fire4: FireModule<B>,
    /// Fire5: 256 -> 384 (squeeze=48, expand=192+192)
    fire5: FireModule<B>,
    /// Fire6: 384 -> 384 (squeeze=48, expand=192+192)
    fire6: FireModule<B>,
    /// Fire7: 384 -> 512 (squeeze=64, expand=256+256)
    fire7: FireModule<B>,
    /// Fire8: 512 -> 512 (squeeze=64, expand=256+256)
    fire8: FireModule<B>,
}

impl<B: Backend> SqueezeFeatureExtractor<B> {
    /// Create a new SqueezeNet feature extractor.
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Conv1: 3 -> 64, 3x3, stride 2
            conv1: Conv2dConfig::new([3, 64], [3, 3])
                .with_stride([2, 2])
                .with_bias(true)
                .init(device),
            // Fire modules (SqueezeNet 1.1 configuration)
            fire1: FireModule::new(64, 16, 64, 64, device),   // -> 128
            fire2: FireModule::new(128, 16, 64, 64, device),  // -> 128
            fire3: FireModule::new(128, 32, 128, 128, device), // -> 256
            fire4: FireModule::new(256, 32, 128, 128, device), // -> 256
            fire5: FireModule::new(256, 48, 192, 192, device), // -> 384
            fire6: FireModule::new(384, 48, 192, 192, device), // -> 384
            fire7: FireModule::new(384, 64, 256, 256, device), // -> 512
            fire8: FireModule::new(512, 64, 256, 256, device), // -> 512
        }
    }

    /// Extract features from 7 SqueezeNet layers.
    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut features = Vec::with_capacity(7);

        // Slice 1: Conv1 + ReLU (64 channels)
        let x = relu(self.conv1.forward(x));
        features.push(x.clone());

        // Slice 2: MaxPool + Fire1 + Fire2 (128 channels)
        let x = max_pool2d_squeeze(x);
        let x = self.fire1.forward(x);
        let x = self.fire2.forward(x);
        features.push(x.clone());

        // Slice 3: MaxPool + Fire3 + Fire4 (256 channels)
        let x = max_pool2d_squeeze(x);
        let x = self.fire3.forward(x);
        let x = self.fire4.forward(x);
        features.push(x.clone());

        // Slice 4: MaxPool + Fire5 (384 channels)
        let x = max_pool2d_squeeze(x);
        let x = self.fire5.forward(x);
        features.push(x.clone());

        // Slice 5: Fire6 (384 channels)
        let x = self.fire6.forward(x);
        features.push(x.clone());

        // Slice 6: Fire7 (512 channels)
        let x = self.fire7.forward(x);
        features.push(x.clone());

        // Slice 7: Fire8 (512 channels)
        let x = self.fire8.forward(x);
        features.push(x);

        features
    }
}

// =============================================================================
// Pooling Helpers
// =============================================================================

/// 2x2 max pooling with stride 2 (for VGG).
fn max_pool2d<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1], false)
}

/// 3x3 max pooling with stride 2 (for AlexNet).
fn max_pool2d_alex<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    burn::tensor::module::max_pool2d(x, [3, 3], [2, 2], [0, 0], [1, 1], false)
}

/// 3x3 max pooling with stride 2, ceil mode (for SqueezeNet).
fn max_pool2d_squeeze<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    burn::tensor::module::max_pool2d(x, [3, 3], [2, 2], [0, 0], [1, 1], true)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{ops::FloatElem, TensorData, Tolerance};

    type FT = FloatElem<TestBackend>;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    // =========================================================================
    // Basic Functionality Tests
    // =========================================================================

    /// Identical images should have LPIPS distance of 0.
    #[test]
    fn test_lpips_identical_images_zero_loss() {
        let device = Default::default();
        let image = TestTensor::<4>::ones([1, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let loss = lpips.forward(image.clone(), image, Reduction::Mean);

        // Identical images → loss = 0
        let expected = TensorData::from([0.0]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    /// Different images should have LPIPS distance greater than 0.
    #[test]
    fn test_lpips_different_images_positive_loss() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let loss = lpips.forward(image1, image2, Reduction::Mean);

        let loss_value = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss_value > 0.0, "LPIPS should be > 0 for different images");
    }

    /// Test symmetry: LPIPS(a, b) == LPIPS(b, a).
    #[test]
    fn test_lpips_symmetry() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let loss_ab = lpips.forward(image1.clone(), image2.clone(), Reduction::Mean);
        let loss_ba = lpips.forward(image2, image1, Reduction::Mean);

        loss_ab
            .into_data()
            .assert_approx_eq::<FT>(&loss_ba.into_data(), Tolerance::default());
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
        let loss = lpips.forward(image1, image2, Reduction::Mean);

        assert_eq!(loss.dims(), [1]);
    }

    #[test]
    fn test_lpips_forward_no_reduction() {
        let device = Default::default();

        let batch_size = 4;
        let image1 = TestTensor::<4>::zeros([batch_size, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([batch_size, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let loss = lpips.forward_no_reduction(image1, image2);

        assert_eq!(loss.dims(), [batch_size]);
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
        let loss = lpips.forward(image1, image2, Reduction::Mean);

        let loss_value = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss_value >= 0.0, "LPIPS should be >= 0");
    }

    // =========================================================================
    // AlexNet Tests
    // =========================================================================

    /// Test AlexNet LPIPS with identical images.
    #[test]
    fn test_lpips_alex_identical_images_zero_loss() {
        let device = Default::default();
        let image = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Alex)
            .init(&device);
        let loss = lpips.forward(image.clone(), image, Reduction::Mean);

        let expected = TensorData::from([0.0]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    /// Test AlexNet LPIPS with different images.
    #[test]
    fn test_lpips_alex_different_images_positive_loss() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 64, 64], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Alex)
            .init(&device);
        let loss = lpips.forward(image1, image2, Reduction::Mean);

        let loss_value = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            loss_value > 0.0,
            "LPIPS (Alex) should be > 0 for different images"
        );
    }

    // =========================================================================
    // SqueezeNet Tests
    // =========================================================================

    /// Test SqueezeNet LPIPS with identical images.
    #[test]
    fn test_lpips_squeeze_identical_images_zero_loss() {
        let device = Default::default();
        let image = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Squeeze)
            .init(&device);
        let loss = lpips.forward(image.clone(), image, Reduction::Mean);

        let expected = TensorData::from([0.0]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    /// Test SqueezeNet LPIPS with different images.
    #[test]
    fn test_lpips_squeeze_different_images_positive_loss() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 64, 64], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Squeeze)
            .init(&device);
        let loss = lpips.forward(image1, image2, Reduction::Mean);

        let loss_value = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            loss_value > 0.0,
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

        let display_str = alloc::format!("{lpips}");
        assert!(display_str.contains("Lpips"));
        assert!(display_str.contains("Vgg"));
    }

    #[test]
    fn display_alex() {
        let device = Default::default();
        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Alex)
            .init(&device);

        let display_str = alloc::format!("{lpips}");
        assert!(display_str.contains("Lpips"));
        assert!(display_str.contains("Alex"));
    }

    #[test]
    fn display_squeeze() {
        let device = Default::default();
        let lpips: Lpips<TestBackend> = LpipsConfig::new()
            .with_net(LpipsNet::Squeeze)
            .init(&device);

        let display_str = alloc::format!("{lpips}");
        assert!(display_str.contains("Lpips"));
        assert!(display_str.contains("Squeeze"));
    }
}

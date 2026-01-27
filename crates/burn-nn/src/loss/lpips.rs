//! LPIPS (Learned Perceptual Image Patch Similarity) loss module.
//!
//! LPIPS measures perceptual similarity between images using deep features.
//! Reference: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
//! https://arxiv.org/abs/1801.03924

use alloc::vec::Vec;

use burn_core as burn;

use super::Reduction;
use crate::PaddingConfig2d;
use crate::conv::{Conv2d, Conv2dConfig};
use burn::config::Config;
use burn::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;

/// VGG network type for LPIPS.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum LpipsNet {
    /// VGG16 network (default)
    Vgg,
    // TODO: impl Alex, Squeeze
    // Alex,
    // Squeeze,
}
/// Configuration for [LPIPS](Lpips) loss module.
///
/// # Example
///
/// ```ignore
/// use burn_nn::loss::{LpipsConfig, LpipsNet};
///
/// let lpips = LpipsConfig::new()
///     .with_net(LpipsNet::Vgg)
///     .with_normalize(true)
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
    /// Initialize a new [LPIPS](Lpips) module.
    ///
    /// # Arguments
    ///
    /// * `device` - Device to create the module on.
    ///
    /// # Returns
    ///
    /// A new LPIPS module with pretrained weights.
    ///
    /// # Note
    ///
    /// Currently only VGG network is supported.
    /// Weights should be loaded from pretrained model for accurate results.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Lpips<B> {
        // VGG16 feature extractor
        let vgg = VggFeatureExtractor::new(device);

        // Linear layers (1x1 conv) for each VGG layer output
        // Channel sizes for VGG16 layers: 64, 128, 256, 512, 512
        let lin0 = Conv2dConfig::new([64, 1], [1, 1])
            .with_bias(false)
            .init(device);
        let lin1 = Conv2dConfig::new([128, 1], [1, 1])
            .with_bias(false)
            .init(device);
        let lin2 = Conv2dConfig::new([256, 1], [1, 1])
            .with_bias(false)
            .init(device);
        let lin3 = Conv2dConfig::new([512, 1], [1, 1])
            .with_bias(false)
            .init(device);
        let lin4 = Conv2dConfig::new([512, 1], [1, 1])
            .with_bias(false)
            .init(device);

        Lpips {
            vgg,
            lin0,
            lin1,
            lin2,
            lin3,
            lin4,
            normalize: self.normalize,
            net: Ignored(self.net),
        }
    }
}

/// LPIPS (Learned Perceptual Image Patch Similarity) loss module.
///
/// Computes perceptual distance between two images using VGG features.
///
/// # Example
///
/// ```ignore
/// use burn_nn::loss::{LpipsConfig, Reduction};
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
    /// VGG feature extractor
    vgg: VggFeatureExtractor<B>,
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

// TODO: Key remapping rules for loading PyTorch lpips AlexNet weights.
// const LPIPS_ALEX_KEY_REMAPS: &[(&str, &str)] = &[
//     // AlexNet layers mapping
//     // ("net\\.slice1\\.0\\.(.*)", "alex.conv1.$1"),
//     // ...
// ];

// TODO: Key remapping rules for loading PyTorch lpips SqueezeNet weights.
// const LPIPS_SQUEEZE_KEY_REMAPS: &[(&str, &str)] = &[
//     // SqueezeNet fire modules mapping
//     // ("net\\.slice1\\.0\\.(.*)", "squeeze.conv1.$1"),
//     // ...
// ];

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
/// // First, save PyTorch lpips weights:
/// // >>> import torch, lpips
/// // >>> loss_fn = lpips.LPIPS(net='vgg')
/// // >>> torch.save(loss_fn.state_dict(), 'lpips_vgg.pt')
///
/// let device = Default::default();
///
/// // Build LoadArgs with all key remappings
/// let mut load_args = LoadArgs::new("lpips_vgg.pt".into());
/// for (pattern, replacement) in lpips_key_remaps(LpipsNet::Vgg) {
///     load_args = load_args.with_key_remap(pattern, replacement);
/// }
///
/// // Load weights
/// let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
///     .load(load_args, &device)
///     .expect("Failed to load weights");
///
/// // Create model and apply weights
/// let lpips = LpipsConfig::new()
///     .init::<Backend>(&device)
///     .load_record(record);
/// ```
pub fn lpips_key_remaps(net: LpipsNet) -> &'static [(&'static str, &'static str)] {
    match net {
        LpipsNet::Vgg => LPIPS_VGG_KEY_REMAPS,
        // LpipsNet::Alex => LPIPS_ALEX_KEY_REMAPS,
        // LpipsNet::Squeeze => LPIPS_SQUEEZE_KEY_REMAPS,
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

        // Extract features from both images
        let feats0 = self.vgg.forward(input);
        let feats1 = self.vgg.forward(target);

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
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, 3, H, W]`
    ///
    /// # Returns
    ///
    /// Vector of 5 feature tensors from each VGG block.
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

/// 2x2 max pooling with stride 2.
fn max_pool2d<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    burn::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1], false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{TensorData, Tolerance, ops::FloatElem};

    type FT = FloatElem<TestBackend>;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    // =========================================================================
    // 기본 기능 테스트 (Basic Functionality Tests)
    // =========================================================================

    /// 동일한 이미지는 LPIPS 거리가 0이어야 함
    #[test]
    fn test_lpips_identical_images_zero_loss() {
        let device = Default::default();
        let image = TestTensor::<4>::ones([1, 3, 32, 32], &device);

        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);
        let loss = lpips.forward(image.clone(), image, Reduction::Mean);

        // 동일 이미지 → loss = 0
        let expected = TensorData::from([0.0]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    /// 다른 이미지는 LPIPS 거리가 0보다 커야 함
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

    /// LPIPS(a, b) == LPIPS(b, a) 대칭성 테스트
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
    // Reduction 테스트
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
    // 출력 범위 테스트
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
    // Display 테스트
    // =========================================================================

    #[test]
    fn display() {
        let device = Default::default();
        let lpips: Lpips<TestBackend> = LpipsConfig::new().init(&device);

        let display_str = alloc::format!("{lpips}");
        assert!(display_str.contains("Lpips"));
    }
}

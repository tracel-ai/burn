//! DISTS (Deep Image Structure and Texture Similarity) metric.
//!
//! DISTS is a full-reference image quality assessment metric that combines
//! structure and texture similarity using deep features from VGG16.
//!
//! Reference: "Image Quality Assessment: Unifying Structure and Texture Similarity"
//! https://arxiv.org/abs/2004.07728

use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay, Param};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_nn::loss::Reduction;

use super::vgg16_l2pool::Vgg16L2PoolExtractor;

/// Channel counts for each stage: [input, stage1, stage2, stage3, stage4, stage5]
const CHANNELS: [usize; 6] = [3, 64, 128, 256, 512, 512];

/// Small constant for numerical stability in structure similarity.
const C1: f32 = 1e-6;

/// Small constant for numerical stability in texture similarity.
const C2: f32 = 1e-6;

/// Configuration for DISTS metric.
#[derive(Config, Debug)]
pub struct DistsConfig {
    /// Whether to normalize input from [0,1] to [-1,1].
    #[config(default = true)]
    pub normalize: bool,
}

impl DistsConfig {
    /// Initialize a DISTS module with default weights.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Dists<B> {
        let total_channels: usize = CHANNELS.iter().sum();

        // Initialize alpha and beta with constant value 0.1 for all channels
        let alpha_data: Vec<f32> = (0..total_channels).map(|_| 0.1).collect();
        let beta_data: Vec<f32> = (0..total_channels).map(|_| 0.1).collect();

        Dists {
            extractor: Vgg16L2PoolExtractor::new(device),
            alpha: Param::from_tensor(Tensor::from_floats(alpha_data.as_slice(), device)),
            beta: Param::from_tensor(Tensor::from_floats(beta_data.as_slice(), device)),
            normalize: self.normalize,
        }
    }

    /// Initialize a DISTS module with pretrained weights.
    pub fn init_pretrained<B: Backend>(&self, device: &B::Device) -> Dists<B> {
        let dists = self.init(device);
        super::weights::load_pretrained_weights(dists)
    }
}

/// DISTS (Deep Image Structure and Texture Similarity) metric.
///
/// Computes perceptual similarity between two images by combining
/// structure similarity (based on spatial means) and texture similarity
/// (based on variances and covariances) across VGG16 feature maps.
///
/// # Example
///
/// ```ignore
/// use burn_train::metric::vision::{DistsConfig, Reduction};
///
/// let device = Default::default();
/// let dists = DistsConfig::new().init_pretrained(&device);
///
/// let img1: Tensor<B, 4> = /* [batch, 3, H, W] */;
/// let img2: Tensor<B, 4> = /* [batch, 3, H, W] */;
///
/// let distance = dists.forward(img1, img2, Reduction::Mean);
/// ```
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Dists<B: Backend> {
    /// VGG16 feature extractor with L2 pooling
    pub(crate) extractor: Vgg16L2PoolExtractor<B>,
    /// Learned weights for structure similarity (per channel)
    pub(crate) alpha: Param<Tensor<B, 1>>,
    /// Learned weights for texture similarity (per channel)
    pub(crate) beta: Param<Tensor<B, 1>>,
    /// Whether to normalize input
    pub(crate) normalize: bool,
}

impl<B: Backend> ModuleDisplay for Dists<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("backbone", &"VGG16-L2Pool".to_string())
            .add("normalize", &self.normalize.to_string())
            .optional()
    }
}

impl<B: Backend> Dists<B> {
    /// Compute DISTS distance with reduction.
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

    /// Compute DISTS distance without reduction.
    ///
    /// # Arguments
    ///
    /// * `input` - First image tensor of shape `[batch, 3, H, W]`
    /// * `target` - Second image tensor of shape `[batch, 3, H, W]`
    ///
    /// # Returns
    ///
    /// Per-sample distance tensor of shape `[batch]`.
    pub fn forward_no_reduction(&self, input: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let [batch, _, _, _] = input.dims();

        // Preprocess inputs
        let (input, target) = self.preprocess(input, target);

        // Extract features from both images
        let feats_x = self.extractor.forward(input);
        let feats_y = self.extractor.forward(target);

        // Get alpha and beta weights
        let alpha = self.alpha.val();
        let beta = self.beta.val();

        // Compute weighted sum of alpha and beta for normalization
        let alpha_sum = alpha.clone().sum();
        let beta_sum = beta.clone().sum();

        let device = feats_x[0].device();

        // Initialize accumulators
        let mut structure_dist = Tensor::<B, 1>::zeros([batch], &device);
        let mut texture_dist = Tensor::<B, 1>::zeros([batch], &device);

        let mut channel_offset = 0;

        // Compute similarity for each stage
        for (feat_x, feat_y) in feats_x.iter().zip(feats_y.iter()) {
            let [_b, c, _h, _w] = feat_x.dims();

            // Get alpha and beta for this stage
            let alpha_stage = alpha.clone().narrow(0, channel_offset, c);
            let beta_stage = beta.clone().narrow(0, channel_offset, c);

            // Compute structure and texture similarity for this stage
            let (s_dist, t_dist) = self.compute_stage_similarity(
                feat_x.clone(),
                feat_y.clone(),
                alpha_stage,
                beta_stage,
            );

            structure_dist = structure_dist.add(s_dist);
            texture_dist = texture_dist.add(t_dist);

            channel_offset += c;
        }

        // Normalize by sum of weights
        structure_dist = structure_dist.div(alpha_sum);
        texture_dist = texture_dist.div(beta_sum);

        // DISTS = 1 - (structure_similarity + texture_similarity)
        // Since we computed distances (1 - similarity), we return the sum
        structure_dist.add(texture_dist)
    }

    /// Compute structure and texture similarity for a single stage.
    fn compute_stage_similarity(
        &self,
        feat_x: Tensor<B, 4>,
        feat_y: Tensor<B, 4>,
        alpha: Tensor<B, 1>,
        beta: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let [batch, channels, height, width] = feat_x.dims();
        let device = feat_x.device();

        // Reshape to [batch, channels, H*W] for easier computation
        let x = feat_x.reshape([batch, channels, height * width]);
        let y = feat_y.reshape([batch, channels, height * width]);

        // Compute means: [batch, channels] (squeeze after mean_dim to remove the reduced dimension)
        let mean_x = x.clone().mean_dim(2).squeeze_dim::<2>(2);
        let mean_y = y.clone().mean_dim(2).squeeze_dim::<2>(2);

        // Compute structure similarity: (2*mean_x*mean_y + c1) / (mean_x^2 + mean_y^2 + c1)
        let c1 = Tensor::<B, 2>::full([batch, channels], C1, &device);
        let structure_sim = mean_x
            .clone()
            .mul(mean_y.clone())
            .mul_scalar(2.0)
            .add(c1.clone())
            .div(
                mean_x
                    .clone()
                    .mul(mean_x.clone())
                    .add(mean_y.clone().mul(mean_y.clone()))
                    .add(c1),
            );

        // Compute variances and covariance
        // var_x = E[x^2] - E[x]^2
        let var_x = x
            .clone()
            .mul(x.clone())
            .mean_dim(2)
            .squeeze_dim::<2>(2)
            .sub(mean_x.clone().mul(mean_x.clone()));
        let var_y = y
            .clone()
            .mul(y.clone())
            .mean_dim(2)
            .squeeze_dim::<2>(2)
            .sub(mean_y.clone().mul(mean_y.clone()));

        // cov_xy = E[xy] - E[x]E[y]
        let cov_xy = x
            .mul(y)
            .mean_dim(2)
            .squeeze_dim::<2>(2)
            .sub(mean_x.clone().mul(mean_y.clone()));

        // Compute texture similarity: (2*cov_xy + c2) / (var_x + var_y + c2)
        let c2 = Tensor::<B, 2>::full([batch, channels], C2, &device);
        let texture_sim = cov_xy
            .mul_scalar(2.0)
            .add(c2.clone())
            .div(var_x.add(var_y).add(c2));

        // Convert similarity to distance: 1 - similarity
        let structure_dist = Tensor::<B, 2>::ones([batch, channels], &device).sub(structure_sim);
        let texture_dist = Tensor::<B, 2>::ones([batch, channels], &device).sub(texture_sim);

        // Apply weights: [batch, channels] * [channels] -> [batch, channels]
        // Then sum over channels -> [batch]
        let alpha_expanded = alpha.unsqueeze_dim::<2>(0).repeat_dim(0, batch);
        let beta_expanded = beta.unsqueeze_dim::<2>(0).repeat_dim(0, batch);

        let weighted_structure = structure_dist
            .mul(alpha_expanded)
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let weighted_texture = texture_dist
            .mul(beta_expanded)
            .sum_dim(1)
            .squeeze_dim::<1>(1);

        (weighted_structure, weighted_texture)
    }

    /// Preprocess input images.
    fn preprocess(
        &self,
        input: Tensor<B, 4>,
        target: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if self.normalize {
            // Normalize from [0, 1] to [-1, 1]
            let input = input.mul_scalar(2.0).sub_scalar(1.0);
            let target = target.mul_scalar(2.0).sub_scalar(1.0);
            (input, target)
        } else {
            (input, target)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::{TensorData, Tolerance, ops::FloatElem};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;
    type FT = FloatElem<TestBackend>;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    #[test]
    fn test_dists_identical_images_zero_distance() {
        let device = Default::default();
        let image = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let dists: Dists<TestBackend> = DistsConfig::new().init(&device);
        let distance = dists.forward(image.clone(), image, Reduction::Mean);

        let expected = TensorData::from([0.0]);
        distance
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_dists_different_images_nonzero_distance() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 64, 64], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 64, 64], &device);

        let dists: Dists<TestBackend> = DistsConfig::new().init(&device);
        let distance = dists.forward(image1, image2, Reduction::Mean);

        let distance_value = distance.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            distance_value.abs() > 1e-6,
            "DISTS should be != 0 for different images"
        );
    }

    #[test]
    fn test_dists_symmetry() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([1, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 32, 32], &device);

        let dists: Dists<TestBackend> = DistsConfig::new().init(&device);
        let distance_forward = dists.forward(image1.clone(), image2.clone(), Reduction::Mean);
        let distance_reverse = dists.forward(image2, image1, Reduction::Mean);

        distance_forward
            .into_data()
            .assert_approx_eq::<FT>(&distance_reverse.into_data(), Tolerance::default());
    }

    #[test]
    fn test_dists_batch_processing() {
        let device = Default::default();

        let image1 = TestTensor::<4>::zeros([2, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([2, 3, 32, 32], &device);

        let dists: Dists<TestBackend> = DistsConfig::new().init(&device);
        let distance = dists.forward(image1, image2, Reduction::Mean);

        assert_eq!(distance.dims(), [1]);
    }

    #[test]
    fn test_dists_no_reduction() {
        let device = Default::default();

        let batch_size = 4;
        let image1 = TestTensor::<4>::zeros([batch_size, 3, 32, 32], &device);
        let image2 = TestTensor::<4>::ones([batch_size, 3, 32, 32], &device);

        let dists: Dists<TestBackend> = DistsConfig::new().init(&device);
        let distance = dists.forward_no_reduction(image1, image2);

        assert_eq!(distance.dims(), [batch_size]);
    }

    #[test]
    fn display_dists() {
        let device = Default::default();
        let dists: Dists<TestBackend> = DistsConfig::new().init(&device);

        let display_str = format!("{dists}");
        assert!(display_str.contains("Dists"));
        assert!(display_str.contains("VGG16-L2Pool"));
    }

    // =========================================================================
    // Pretrained Weights Tests (requires network)
    // =========================================================================

    /// Test DISTS pretrained weights download and loading.
    #[test]
    fn test_dists_pretrained() {
        let device = Default::default();

        let dists: Dists<TestBackend> = DistsConfig::new().init_pretrained(&device);

        // Test with identical images - should be ~0
        let image = TestTensor::<4>::ones([1, 3, 64, 64], &device);
        let distance = dists.forward(image.clone(), image, Reduction::Mean);
        let distance_value = distance.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            distance_value.abs() < 1e-5,
            "Pretrained DISTS should be ~0 for identical images, got {}",
            distance_value
        );

        // Test with different images - should be positive
        let image1 = TestTensor::<4>::zeros([1, 3, 64, 64], &device);
        let image2 = TestTensor::<4>::ones([1, 3, 64, 64], &device);
        let distance = dists.forward(image1, image2, Reduction::Mean);
        let distance_value = distance.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            distance_value > 0.0,
            "Pretrained DISTS should be > 0 for different images"
        );
    }
}

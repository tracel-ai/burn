use crate::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
    state::{FormatOptions, NumericMetricState},
};
use burn_core::{
    prelude::{Backend, Int, Tensor},
    tensor::{
        ElementConversion,
        module::{avg_pool2d, conv2d},
        ops::{ConvOptions, PadMode},
    },
};
use core::marker::PhantomData;

/// Input type for the [MsSsimMetric].
///
/// Both tensors must have shape `[N, C, H, W]`:
/// - `N`: Batch size
/// - `C`: Number of channels (1 for grayscale, 3 for RGB, etc.)
/// - `H`: Height
/// - `W`: Width
///
/// # Important
/// The image dimensions must be sufficiently large to accommodate the multi-scale
/// computation. Specifically, for the default 5 scales used by Burn, the image dimensions
/// should be at least `kernel_size * 2^(scales-1)` (e.g., 11 × 2^4 = 11 * 16 = 176 for default kernel size).
/// If your images are smaller, reduce the kernel size or number of scales.
///
/// # Example
/// ```rust,ignore
/// // Create input for RGB images
/// let outputs: Tensor<B, 4> = /* tensor */;
/// let targets: Tensor<B, 4> = /* tensor */;
/// let input = MsSsimInput::new(outputs, targets);
/// ```
pub struct MsSsimInput<B: Backend> {
    /// Model outputs with shape [N, C, H, W].
    outputs: Tensor<B, 4>,
    /// Ground truth targets with shape [N, C, H, W].
    targets: Tensor<B, 4>,
}

impl<B: Backend> MsSsimInput<B> {
    /// Creates a new MsSsimInput with the given outputs and targets.
    ///
    /// # Arguments
    /// - `outputs`: The model output images with shape [N, C, H, W].
    /// - `targets`: The ground truth images with shape [N, C, H, W].
    ///
    /// # Returns
    /// A new instance of `MsSsimInput`.
    ///
    /// # Panics
    /// - If `outputs` and `targets` do not have the same shape.
    pub fn new(outputs: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
        assert!(
            outputs.dims() == targets.dims(),
            "Shape mismatch: outputs {:?} targets {:?}",
            outputs.dims(),
            targets.dims()
        );
        Self { outputs, targets }
    }
}

/// Configuration for the [MsSsimMetric].
#[derive(Debug, Clone)]
pub struct MsSsimMetricConfig {
    /// A parameter of SSIM used to stabilize the luminance comparison.
    /// Default is 0.01.
    pub k1: f32,
    /// A parameter of SSIM used to stabilize the contrast comparison.
    /// Default is 0.03.
    pub k2: f32,
    /// The range of the pixel values in images which can be computed as following:
    /// `let pixel_range = max_pixel_val - min_pixel_val;`
    /// where `max_pixel_val` is the maximum possible pixel value and `min_pixel_val`
    /// is the minimum possible pixel value.
    ///
    /// - For normalized images in range [0, 1], it should be set to `1.0 - 0.0 = 1.0`
    /// - For normalized images in range [-1, 1], it should be set to `1.0 - (-1.0) = 2.0`
    /// - For 8-bit images in range [0, 255], it should be set to `255.0 - 0.0 = 255.0`
    pub pixel_range: f32,
    /// The MS-SSIM metric involves applying convolution to the input tensors using a Gaussian kernel.
    /// This is the kernel size of the Gaussian kernel. Default is 11.
    pub kernel_size: usize,
    /// The MS-SSIM metric involves applying convolution to the input tensors using a Gaussian kernel.
    /// This is the standard deviation of the Gaussian kernel. Default is 1.5.
    pub sigma: f32,
    /// The number of channels in the input images (e.g., 1 for grayscale, 3 for RGB).
    /// This is used to create the appropriate convolution kernels. Default is 3.
    pub channels: usize,
    /// The weights/betas for each scale in the MS-SSIM computation.
    /// The length of this vector determines the number of scales.
    /// Default is \[0.0448, 0.2856, 0.3001, 0.2363, 0.1333\] (5 scales).
    pub betas: Vec<f32>,
}

impl MsSsimMetricConfig {
    /// Creates a configuration with the specified data range and default parameters.
    ///
    /// # Default parameters
    /// - k1: 0.01
    /// - k2: 0.03
    /// - kernel_size: 11
    /// - sigma: 1.5
    /// - channels: 3
    ///
    /// # Panics
    /// - If `pixel_range` is not positive.
    ///
    /// # Example
    /// ```rust,ignore
    /// // For normalized RGB images [0, 1]
    /// let config1 = MsSsimMetricConfig::new(1.0);
    ///
    /// // For 8-bit images [0, 255]  
    /// let config2 = MsSsimMetricConfig::new(255.0);
    ///
    /// // For grayscale with custom kernel
    /// let config3 = MsSsimMetricConfig::new(1.0)
    ///     .with_channels(1)
    ///     .with_kernel_size(7);
    /// ```
    pub fn new(pixel_range: f32) -> Self {
        assert!(pixel_range > 0.0, "pixel_range must be positive");
        Self {
            k1: 0.01,
            k2: 0.03,
            pixel_range,
            kernel_size: 11,
            sigma: 1.5,
            channels: 3,
            betas: vec![0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        }
    }

    /// Sets custom values for the k1 and k2 parameters of MS-SSIM which are
    /// used for numerical stability.
    ///
    /// # Default values
    /// - k1: 0.01
    /// - k2: 0.03
    ///
    /// # Panics
    /// - If `k1` or `k2` is not positive.
    pub fn with_k1_k2(mut self, k1: f32, k2: f32) -> Self {
        assert!(k1 > 0.0, "k1 must be positive");
        assert!(k2 > 0.0, "k2 must be positive");
        self.k1 = k1;
        self.k2 = k2;
        self
    }

    /// Sets a custom kernel size for the Gaussian kernel used in MS-SSIM. The
    /// kernel size must be a positive odd number.
    ///
    /// # Default value
    /// - kernel_size: 11
    ///
    /// # Panics
    /// - If `kernel_size` is not a positive odd number.
    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        assert!(
            kernel_size > 0 && kernel_size % 2 == 1,
            "kernel_size must be positive and an odd number"
        );
        self.kernel_size = kernel_size;
        self
    }

    /// Sets a custom sigma (standard deviation) for the Gaussian kernel used in MS-SSIM.
    ///
    /// # Default value
    /// - sigma: 1.5
    ///
    /// # Panics
    /// - If `sigma` is not positive.
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        assert!(sigma > 0.0, "sigma must be a positive number");
        self.sigma = sigma;
        self
    }

    /// Sets the number of channels for the input images.
    ///
    /// This affects the shape of the pre-computed convolution kernels.
    /// Change this if working with grayscale (1) or multispectral images (>3).
    ///
    /// # Default value
    /// - channels: 3
    ///
    /// # Panics
    /// - If `channels` is 0.
    pub fn with_channels(mut self, channels: usize) -> Self {
        assert!(channels > 0, "channels must be a positive number");
        self.channels = channels;
        self
    }

    /// Sets custom betas for the scales. The length of the betas vector
    /// determines the number of scales used in the MS-SSIM computation.
    ///
    /// # Default value
    /// - betas: `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]` (5 scales)
    ///
    /// # Panics
    /// - If `betas` is empty.
    pub fn with_betas(mut self, betas: Vec<f32>) -> Self {
        assert!(!betas.is_empty(), "betas vector cannot be empty");

        assert!(
            betas.iter().all(|&b| b >= 0.0),
            "All beta values must be non-negative"
        );

        let sum: f32 = betas.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "The sum of the betas must be 1.0, but got {}",
            sum
        );

        self.betas = betas;
        self
    }
}

/// Multi-Scale Structural Similarity Index (MS-SSIM) metric for image quality assessment.
///
/// MS-SSIM extends the single-scale [SSIM](crate::metric::vision::SsimMetric) by computing
/// the index at multiple resolutions (scales) and combining them using weighted averaging.
/// This approach better correlates with human visual perception, especially for
/// high-resolution images where fine details and texture variations are important.
///
/// # Algorithm Overview
///
/// MS-SSIM computes structural similarity across M scales (M=5 in Burn):
///
/// 1. **Contrast** and **Structure** components are computed at every scale
/// 2. **Luminance** is computed only at the coarsest (last) scale
/// 3. Between scales, images are downsampled by a factor of 2 using average pooling
///
/// The final metric is computed as:
/// ```text
/// MS-SSIM = L_M^{α_M} × ∏_{j=1}^M (C_j^{β_j} × S_j^{γ_j})
/// ```
///
/// Where:
/// - `L_M` is luminance at the last scale (M)
/// - `C_j` is contrast at scale j: `(2σ_xσ_y + C2) / (σ_x² + σ_y² + C2)`
/// - `S_j` is structure at scale j: `(σ_xy + C3) / (σ_xσ_y + C3)`
/// - `α_M, β_j, γ_j` are weights from Wang et al. (\[0.0448, 0.2856, 0.3001, 0.2363, 0.1333\])
///
/// # Notes
///
/// - This implementation uses separable Gaussian convolution for efficiency (reduces complexity from O(K^2) to O(2K) per pixel)
/// - Gaussian kernels are pre-computed during initialization to avoid redundant computation
/// - The metric requires images to be large enough to survive the downsampling operations
///
/// # Value Range
///
/// MS-SSIM values typically range from 0 to 1, where:
/// - 1.0 indicates perfect structural similarity (identical images)
/// - 0.0 indicates no structural similarity
/// - Values are usually positive due to the stability constants (C1, C2, C3)
///
/// # References
///
/// [Multi-scale Structural Similarity for Image Quality Assessment](https://www.cns.nyu.edu/pub/eero/wang03b.pdf)
#[derive(Clone)]
pub struct MsSsimMetric<B: Backend> {
    name: MetricName,
    /// Internal state for numeric metric aggregation.
    state: NumericMetricState,
    /// Marker for backend type.
    _b: PhantomData<B>,
    /// Configuration for the metric.
    config: MsSsimMetricConfig,
    /// Pre-computed horizontal Gaussian kernel with shape [C, 1, 1, K]
    horizontal_kernel: Tensor<B, 4>,
    /// Pre-computed vertical Gaussian kernel with shape [C, 1, K, 1]
    vertical_kernel: Tensor<B, 4>,
}

impl<B: Backend> MsSsimMetric<B> {
    /// Creates a new MS-SSIM metric with the given configuration.
    ///
    /// # Arguments
    /// - `config`: Configuration for the metric (data range, kernel size, etc.)
    /// - `device`: Device to place the Gaussian kernels on
    ///
    /// # Note
    /// The default metric name format is "MS-SSIM (pr={}, k={}, σ={})"
    /// where pr is the pixel range, k is the kernel size, and σ is the
    /// standard deviation.
    ///
    /// # Example
    /// ```ignore
    /// let config = MsSsimMetricConfig::new(1.0).with_channels(1); // Grayscale
    /// let metric = MsSsimMetric::<B>::new(config, &device);
    /// ```
    pub fn new(config: MsSsimMetricConfig, device: &B::Device) -> Self {
        let kernel = Self::create_1d_gaussian_kernel(&config, device);
        let size = config.kernel_size;

        // Create horizontal kernel: shape [C, 1, 1, K] for depthwise conv
        let horizontal_kernel = kernel
            .clone()
            .reshape([1, 1, 1, size])
            .repeat_dim(0, config.channels);

        // Create vertical kernel: shape [C, 1, K, 1] for depthwise conv
        let vertical_kernel = kernel
            .reshape([1, 1, size, 1])
            .repeat_dim(0, config.channels);

        Self {
            name: MetricName::new(format!(
                "MS-SSIM (pr={}, k={}, σ={})",
                config.pixel_range, config.kernel_size, config.sigma
            )),
            state: NumericMetricState::default(),
            _b: PhantomData,
            config,
            horizontal_kernel,
            vertical_kernel,
        }
    }

    /// Overrides the default metric name.
    ///
    /// # Example
    /// ```ignore
    /// let metric = MsSsimMetric::<B>::new(config, &device)
    ///     .with_name("Custom MS-SSIM Name");
    /// ```
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = MetricName::new(name.to_string());
        self
    }

    /// Creates a normalized 1D Gaussian kernel as a tensor where the kernel values sum to 1.0.
    fn create_1d_gaussian_kernel(config: &MsSsimMetricConfig, device: &B::Device) -> Tensor<B, 1> {
        let size = config.kernel_size as i64;
        let sigma = config.sigma;
        let center = (size / 2) as f32;

        let one_to_size_tensor = Tensor::<B, 1, Int>::arange(0..size, device).float();
        let x_vals = one_to_size_tensor.sub_scalar(center);

        // Gaussian: exp(-x² / 2σ²)
        let x_squared = x_vals.clone().mul(x_vals);
        let x_squared_div_2_sigma_squared = x_squared.div_scalar(2.0 * sigma * sigma);
        let unnormalized_kernel = x_squared_div_2_sigma_squared.neg().exp();
        let kernel_vals_sum = unnormalized_kernel.clone().sum();
        unnormalized_kernel.div(kernel_vals_sum)
    }

    /// Applies separable Gaussian convolution using pre-computed kernels.
    ///
    /// Performs two 1D convolutions (horizontal then vertical) which is
    /// computationally cheaper than a single 2D convolution.
    ///
    /// # Arguments
    /// - `input`: Tensor of shape [N, C, H, W]
    fn gaussian_separable_conv(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let padding = self.config.kernel_size / 2;
        let h_kernel = self.horizontal_kernel.clone();
        let v_kernel = self.vertical_kernel.clone();

        // Apply reflect padding to all 4 sides of the input tensor before convolution
        // Format: (left, right, top, bottom)
        let padded_input = input.pad((padding, padding, padding, padding), PadMode::Reflect);

        let h_conv_options = ConvOptions::new([1, 1], [0, 0], [1, 1], self.config.channels);
        let v_conv_options = ConvOptions::new([1, 1], [0, 0], [1, 1], self.config.channels);

        let input_after_h_conv = conv2d(padded_input, h_kernel, None, h_conv_options);
        conv2d(input_after_h_conv, v_kernel, None, v_conv_options)
    }
}

impl<B: Backend> Metric for MsSsimMetric<B> {
    type Input = MsSsimInput<B>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let dims = item.outputs.dims();
        let scales = self.config.betas.len();

        assert_eq!(
            dims[1], self.config.channels,
            "Input has {} channels but metric was configured for {}",
            dims[1], self.config.channels
        );

        // Verify minimum size for the given number of scales
        // After (scales - 1) downsamples, size is original / 2^(scales-1)
        // We need kernel_size at that scale
        let downsample_ops_num = scales.saturating_sub(1) as u32;
        let min_size = self.config.kernel_size * (2usize.pow(downsample_ops_num));
        let h = dims[2];
        let w = dims[3];
        assert!(
            h >= min_size && w >= min_size,
            "Image dimensions (H={}, W={}) must be at least {} to support {} scales of MS-SSIM \
                with kernel_size={}. Either increase image size, reduce kernel_size, or reduce the number of scales (betas).",
            h,
            w,
            min_size,
            scales,
            self.config.kernel_size
        );

        let mut x = item.outputs.clone();
        let mut y = item.targets.clone();
        let betas = &self.config.betas;

        // Compute c1 = (k1 * L)^2 and c2 = (k2 * L)^2, c3 = c2/2
        let c1 = (self.config.k1 * self.config.pixel_range).powi(2);
        let c2 = (self.config.k2 * self.config.pixel_range).powi(2);

        // Initialize accumulator to 1 for update via multiplication
        // Shape: [N, C]
        let batch_size = dims[0];
        let channels = dims[1];
        let mut ms_ssim_tensor = Tensor::<B, 2>::ones([batch_size, channels], &item.outputs.device());

        for i in 0..betas.len() {
            // Compute mu_x and mu_y
            let mu_x = self.gaussian_separable_conv(x.clone());
            let mu_y = self.gaussian_separable_conv(y.clone());
            let square_of_mu_x = mu_x.clone() * mu_x.clone();
            let square_of_mu_y = mu_y.clone() * mu_y.clone();

            // Var(X) = E(X^2) - E(X)^2
            let mu_of_x_squared = self.gaussian_separable_conv(x.clone() * x.clone());
            let mu_of_y_squared = self.gaussian_separable_conv(y.clone() * y.clone());
            let var_x = (mu_of_x_squared - square_of_mu_x.clone()).clamp_min(0.0);
            let var_y = (mu_of_y_squared - square_of_mu_y.clone()).clamp_min(0.0);

            // Cov(X, Y) = E(XY) - E(X)E(Y)
            let mu_of_xy = self.gaussian_separable_conv(x.clone() * y.clone());
            let cov_xy = mu_of_xy - (mu_x.clone() * mu_y.clone());

            // Compute cs_map = (2σxy + C2) / (σx² + σy² + C2)
            // This is mathematically equivalent to c(x,y) * s(x,y) when C3 = C2 / 2
            let contrast_structure = (cov_xy * 2.0 + c2) / (var_x + var_y + c2);
            let beta_j = betas[i];

            // Include luminance at the last scale
            if i == betas.len() - 1 {
                // Compute l(x, y) = (2μxμy + C1) / (μx² + μy² + C1)
                let luminance: Tensor<B, 4> =
                    (2 * mu_x * mu_y + c1) / (square_of_mu_x + square_of_mu_y + c1);
                let ssim = luminance * contrast_structure;
                let ssim_spatial_mean = ssim.mean_dims(&[2, 3]).reshape([batch_size, channels]);
                // Clamp to avoid negative values before raising to power (prevents NaNs)
                let ssim_mean_clamped = ssim_spatial_mean.clamp_min(0.0);
                ms_ssim_tensor = ms_ssim_tensor * ssim_mean_clamped.powf_scalar(beta_j);
            } else {
                let contrast_structure_spatial_mean = contrast_structure
                    .mean_dims(&[2, 3])
                    .reshape([batch_size, channels]);
                // Clamp to avoid negative values before raising to power (prevents NaNs)
                let c_s_mean_clamped = contrast_structure_spatial_mean.clamp_min(0.0);
                ms_ssim_tensor = ms_ssim_tensor * c_s_mean_clamped.powf_scalar(beta_j);

                x = avg_pool2d(x, [2, 2], [2, 2], [0, 0], false, false);
                y = avg_pool2d(y, [2, 2], [2, 2], [0, 0], false, false);
            }
        }

        let ms_ssim_per_image = ms_ssim_tensor.mean_dim(1);
        let avg_ms_ssim = ms_ssim_per_image.mean().into_scalar().elem::<f64>();

        self.state.update(
            avg_ms_ssim,
            batch_size,
            FormatOptions::new(self.name()).precision(4),
        )
    }

    /// Clears the metric state.
    fn clear(&mut self) {
        self.state.reset();
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Numeric for MsSsimMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestBackend, metric::Numeric};
    use burn_core::tensor::Distribution;

    fn test_config() -> MsSsimMetricConfig {
        // Use small kernel and single channel for testing
        // With kernel_size=3, we need images >= 3*16=48
        MsSsimMetricConfig::new(1.0)
            .with_kernel_size(3)
            .with_sigma(1.0)
            .with_channels(1)
    }

    #[test]
    fn test_ms_ssim_perfect_similarity() {
        // Identical images should give MS-SSIM = 1.0
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            [[[
                [0.5_f32; 64]; 64  // 64x64 constant image
            ]]],
            &device,
        );
        let targets = outputs.clone();

        let mut metric = MsSsimMetric::<TestBackend>::new(test_config(), &device);
        let input = MsSsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ms_ssim = metric.value().current();
        assert!(
            ms_ssim > 0.99,
            "MS-SSIM for identical images should be 1.0, got {}",
            ms_ssim
        );
    }

    #[test]
    fn test_ms_ssim_completely_different() {
        // Black vs white images should give very low MS-SSIM (close to 0.0)
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::zeros([1, 1, 256, 256], &device);
        let targets = Tensor::<TestBackend, 4>::ones([1, 1, 256, 256], &device);

        let mut metric = MsSsimMetric::<TestBackend>::new(test_config(), &device);
        let input = MsSsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ms_ssim = metric.value().current();
        assert!(
            (ms_ssim - 0.3).abs() < 0.01,
            "MS-SSIM for black vs white should be low (around 0.3), got {}",
            ms_ssim
        );
    }

    #[test]
    fn test_ms_ssim_similar_images() {
        // Small perturbation should give high MS-SSIM (close to 1.0)
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::full([1, 1, 64, 64], 0.5, &device);
        let targets = Tensor::<TestBackend, 4>::full([1, 1, 64, 64], 0.52, &device);

        let mut metric = MsSsimMetric::<TestBackend>::new(test_config(), &device);
        let input = MsSsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ms_ssim = metric.value().current();
        assert!(
            ms_ssim > 0.95,
            "MS-SSIM for very similar images should be close to 1.0, got {}",
            ms_ssim
        );
    }

    #[test]
    fn test_ms_ssim_batch_averaging() {
        let device = Default::default();
        // Batch of 2: one identical, one different
        let outputs = Tensor::<TestBackend, 4>::from_data(
            [
                [[[0.5_f32; 64]; 64]], // Image 1: constant 0.5
                [[[0.0_f32; 64]; 64]], // Image 2: constant 0.0 (black)
            ],
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::from_data(
            [
                [[[0.5_f32; 64]; 64]], // Image 1: identical
                [[[1.0_f32; 64]; 64]], // Image 2: white (opposite)
            ],
            &device,
        );

        let mut metric = MsSsimMetric::<TestBackend>::new(test_config(), &device);
        let input = MsSsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ms_ssim = metric.value().current();
        // Average of ~1.0 and ~0.292 should be around 0.64
        assert!(
            (ms_ssim - 0.64).abs() < 0.02,
            "Average MS-SSIM should be around 0.64, got {}",
            ms_ssim
        );
    }

    #[test]
    fn test_ms_ssim_multichannel() {
        let device = Default::default();
        // Test with 3 channels (RGB)
        let config = MsSsimMetricConfig::new(1.0)
            .with_kernel_size(3)
            .with_sigma(1.0)
            .with_channels(3);

        let outputs = Tensor::<TestBackend, 4>::random(
            [2, 3, 64, 64],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let targets = outputs.clone();

        let mut metric = MsSsimMetric::<TestBackend>::new(config, &device);
        let input = MsSsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ms_ssim = metric.value().current();
        assert!(
            ms_ssim > 0.99,
            "MS-SSIM for identical RGB images should be 1.0, got {}",
            ms_ssim
        );
    }

    #[test]
    fn test_ms_ssim_running_average() {
        let device = Default::default();
        let mut metric = MsSsimMetric::<TestBackend>::new(test_config(), &device);

        // First update: identical (1.0)
        let img1 = Tensor::<TestBackend, 4>::full([1, 1, 64, 64], 0.5, &device);
        let input1 = MsSsimInput::new(img1.clone(), img1);
        metric.update(&input1, &MetricMetadata::fake());

        assert!(
            metric.value().current() > 0.99,
            "First update should be approximately 1.0"
        );

        // Second update: different (~0.29)
        let black = Tensor::<TestBackend, 4>::zeros([1, 1, 64, 64], &device);
        let white = Tensor::<TestBackend, 4>::ones([1, 1, 64, 64], &device);
        let input2 = MsSsimInput::new(black, white);
        metric.update(&input2, &MetricMetadata::fake());

        let running = metric.running_value().current();
        assert!(
            (running - 0.64).abs() < 0.02,
            "Running average should be approximately 0.64, got {}",
            running
        );
    }

    #[test]
    fn test_ms_ssim_single_scale_small_image() {
        let device = Default::default();
        // Default 5 scales with kernel_size=11 requires a 176x176 image.
        // With a single scale, the minimum required size drops to
        // just 11x11 (kernel_size * 2^0).
        let config = MsSsimMetricConfig::new(1.0)
            .with_channels(1)
            .with_betas(vec![1.0]); // 1 scale

        let mut metric = MsSsimMetric::<TestBackend>::new(config, &device);

        // Create a 16x16 image. This would normally panic with 5 scales,
        // but should succeed with 1 scale.
        let outputs = Tensor::<TestBackend, 4>::zeros([1, 1, 16, 16], &device);
        let targets = outputs.clone();
        let input = MsSsimInput::new(outputs, targets);

        // This should not panic
        let _ = metric.update(&input, &MetricMetadata::fake());

        // Identical images should still yield ~1.0
        let ms_ssim = metric.value().current();
        assert!(
            ms_ssim > 0.99,
            "1-scale MS-SSIM for identical images should be 1.0, got {}",
            ms_ssim
        );
    }

    #[test]
    fn test_ssim_symmetry() {
        // MS-SSIM(x, y) should equal MS-SSIM(y, x)
        // Symmetry is one of the mathematical properties of MS-SSIM
        let device = Default::default();
        let config = MsSsimMetricConfig::new(1.0)
            .with_kernel_size(3)
            .with_sigma(1.0)
            .with_channels(3);

        let img1 = Tensor::<TestBackend, 4>::random(
            [2, 3, 64, 64],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let img2 = Tensor::<TestBackend, 4>::random(
            [2, 3, 64, 64],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let mut metric1 = MsSsimMetric::<TestBackend>::new(config.clone(), &device);
        let input1 = MsSsimInput::new(img1.clone(), img2.clone());
        let _entry = metric1.update(&input1, &MetricMetadata::fake());
        let ms_ssim1 = metric1.value().current();

        let mut metric2 = MsSsimMetric::<TestBackend>::new(config, &device);
        let input2 = MsSsimInput::new(img2, img1);
        let _entry = metric2.update(&input2, &MetricMetadata::fake());
        let ms_ssim2 = metric2.value().current();

        assert!(
            (ms_ssim1 - ms_ssim2).abs() < 0.001,
            "MS-SSIM should be symmetric: MS-SSIM(x,y)={} vs MS-SSIM(y,x)={}",
            ms_ssim1,
            ms_ssim2
        );
    }

    #[test]
    fn test_ms_ssim_clear() {
        let device = Default::default();
        let mut metric = MsSsimMetric::<TestBackend>::new(test_config(), &device);

        let img = Tensor::<TestBackend, 4>::full([1, 1, 64, 64], 0.5, &device);
        let input = MsSsimInput::new(img.clone(), img);
        metric.update(&input, &MetricMetadata::fake());

        assert!(metric.value().current() > 0.99);

        metric.clear();
        assert!(metric.running_value().current().is_nan());
    }

    #[test]
    fn test_ms_ssim_custom_name() {
        let device = Default::default();
        let config = MsSsimMetricConfig::new(1.0);
        let metric = MsSsimMetric::<TestBackend>::new(config, &device).with_name("CustomMS-SSIM");
        assert_eq!(metric.name().to_string(), "CustomMS-SSIM");
    }

    #[test]
    fn test_ms_ssim_default_name() {
        let device = Default::default();
        let config = MsSsimMetricConfig::new(255.0);
        let metric = MsSsimMetric::<TestBackend>::new(config, &device);
        assert_eq!(metric.name().to_string(), "MS-SSIM (pr=255, k=11, σ=1.5)");
    }

    #[test]
    fn test_ms_ssim_attributes() {
        let device = Default::default();
        let config = MsSsimMetricConfig::new(1.0);
        let metric = MsSsimMetric::<TestBackend>::new(config, &device);

        match metric.attributes() {
            MetricAttributes::Numeric(attrs) => {
                assert!(attrs.higher_is_better);
                assert_eq!(attrs.unit, None);
            }
            _ => panic!("Expected numeric attributes"),
        }
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_ms_ssim_shape_mismatch() {
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::zeros([1, 1, 64, 64], &device);
        let targets = Tensor::<TestBackend, 4>::zeros([1, 1, 32, 32], &device);
        let _ = MsSsimInput::new(outputs, targets);
    }

    #[test]
    #[should_panic(expected = "k1 must be positive")]
    fn test_ms_ssim_negative_k1() {
        let _ = MsSsimMetricConfig::new(1.0).with_k1_k2(-0.01, 0.03);
    }

    #[test]
    #[should_panic(expected = "k2 must be positive")]
    fn test_ms_ssim_negative_k2() {
        let _ = MsSsimMetricConfig::new(1.0).with_k1_k2(0.01, -0.03);
    }

    #[test]
    #[should_panic(expected = "pixel_range must be positive")]
    fn test_ms_ssim_negative_data_range() {
        let _ = MsSsimMetricConfig::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "pixel_range must be positive")]
    fn test_ms_ssim_zero_data_range() {
        let _ = MsSsimMetricConfig::new(0.0);
    }

    #[test]
    #[should_panic(expected = "kernel_size must be positive and an odd number")]
    fn test_ms_ssim_even_kernel_size() {
        let _ = MsSsimMetricConfig::new(1.0).with_kernel_size(10);
    }

    #[test]
    #[should_panic(expected = "kernel_size must be positive and an odd number")]
    fn test_ms_ssim_zero_kernel_size() {
        let _ = MsSsimMetricConfig::new(1.0).with_kernel_size(0);
    }

    #[test]
    #[should_panic(expected = "sigma must be a positive number")]
    fn test_ms_ssim_negative_sigma() {
        let _ = MsSsimMetricConfig::new(1.0).with_sigma(-1.5);
    }

    #[test]
    #[should_panic(expected = "sigma must be a positive number")]
    fn test_ms_ssim_zero_sigma() {
        let _ = MsSsimMetricConfig::new(1.0).with_sigma(0.0);
    }

    #[test]
    #[should_panic(expected = "channels must be a positive number")]
    fn test_ms_ssim_zero_channels() {
        let _ = MsSsimMetricConfig::new(1.0).with_channels(0);
    }

    #[test]
    #[should_panic(expected = "betas vector cannot be empty")]
    fn test_ms_ssim_empty_betas() {
        let _ = MsSsimMetricConfig::new(1.0).with_betas(vec![]);
    }

    #[test]
    #[should_panic(expected = "All beta values must be non-negative")]
    fn test_ms_ssim_negative_betas() {
        let _ = MsSsimMetricConfig::new(1.0).with_betas(vec![0.3, 0.3, -0.1, 0.5]);
    }

    #[test]
    #[should_panic(expected = "The sum of the betas must be 1.0")]
    fn test_ms_ssim_invalid_betas_sum() {
        let _ = MsSsimMetricConfig::new(1.0).with_betas(vec![0.5, 0.3]);
    }

    #[test]
    #[should_panic(expected = "Image dimensions")]
    fn test_ms_ssim_image_too_small() {
        let device = Default::default();
        // 3 scales with kernel_size=11 requires 44x44 minimum (11 * 2^2)
        let config = MsSsimMetricConfig::new(1.0).with_betas(vec![0.5, 0.3, 0.2]);
        let mut metric = MsSsimMetric::<TestBackend>::new(config, &device);

        let outputs = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device); // Too small (32 < 44)
        let targets = outputs.clone();
        let input = MsSsimInput::new(outputs, targets);
        let _ = metric.update(&input, &MetricMetadata::fake());
    }
}

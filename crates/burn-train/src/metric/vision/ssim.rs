use crate::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
    state::{FormatOptions, NumericMetricState},
};
use burn_core::{
    prelude::{Backend, Tensor},
    tensor::{ElementConversion, module::conv2d, ops::ConvOptions},
};
use core::marker::PhantomData;

/// Input type for the [SsimMetric].
///
/// Both tensors must have shape `[N, C, H, W]`:
/// - `N`: Batch size
/// - `C`: Number of channels (1 for grayscale, 3 for RGB, etc.)
/// - `H`: Height
/// - `W`: Width
pub struct SsimInput<B: Backend> {
    /// Model output (predictions/reconstructions) images with shape [N, C, H, W].
    outputs: Tensor<B, 4>,
    /// Ground truth images with shape [N, C, H, W].
    targets: Tensor<B, 4>,
}

impl<B: Backend> SsimInput<B> {
    /// Creates a new SsimInput with the given outputs and targets.
    ///
    /// Inputs are expected to have the dimensions `[N, C, H, W]`
    /// where `N` is the batch size, `C` is the number of channels,
    /// `H` is the height of the image, and `W` is the width of the image.
    ///
    /// # Arguments
    /// - `outputs`: The model output images with shape [N, C, H, W].
    /// - `targets`: The ground truth images with shape [N, C, H, W].
    ///
    /// # Returns
    /// A new instance of `SsimInput`.
    ///
    /// # Panics
    /// - If `outputs` and `targets` do not have the same shape.
    pub fn new(outputs: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
        assert!(
            outputs.dims() == targets.dims(),
            "Shape mismatch: outputs {:?}, targets {:?}",
            outputs.dims(),
            targets.dims()
        );
        Self { outputs, targets }
    }
}

/// Configuration for the [SsimMetric].
#[derive(Debug, Clone, Copy)]
pub struct SsimMetricConfig {
    /// The range of the pixel values in images which can be computed as following:
    /// `let data_range = max_pixel_val - min_pixel_val;`
    /// where `max_pixel_val` is the maximum possible pixel value and `min_pixel_val`
    /// is the minimum possible pixel value.
    ///
    /// - For normalized images in range [0, 1], it should be set to `1.0 - 0.0 = 1.0`
    /// - For normalized images in range [-1, 1], it should be set to `1.0 - (-1.0) = 2.0`
    /// - For 8-bit images in range [0, 255], it should be set to `255.0 - 0.0 = 255.0`
    pub data_range: f64,
    /// A parameter of SSIM used to stabilize the luminance comparison.
    /// Default is 0.01.
    pub k1: f64,
    /// A parameter of SSIM used to stabilize the contrast comparison.
    /// Default is 0.03.
    pub k2: f64,
    /// The SSIM metric involves applying convolution to the input tensors using a Gaussian kernel.
    /// This is the window/kernel size of the Gaussian kernel. Default is 11.
    pub window_size: usize,
    /// The SSIM metric involves applying convolution to the input tensors using a Gaussian kernel.
    /// This is the standard deviation of the Gaussian kernel. Default is 1.5.
    pub sigma: f64,
}

impl SsimMetricConfig {
    /// Creates a configuration with the specified data range and default parameters.
    ///
    /// # Default parameters
    /// - k1: 0.01
    /// - k2: 0.03
    /// - window_size: 11
    /// - sigma: 1.5
    ///
    /// # Panics
    /// - If `data_range` is not positive.
    ///
    /// # Example
    /// ```ignore
    /// // Normalized images [0, 1]
    /// let config1 = SsimMetricConfig::new(1.0);
    ///
    /// // 8-bit images [0, 255]  
    /// let config2 = SsimMetricConfig::new(255.0);
    ///
    /// // Also set custom values for k1 and k2
    /// let config3 = SsimMetricConfig::new(1.0).with_k1_k2(0.015, 0.025);
    ///
    /// // Also set a custom value for window size
    /// config3.with_window_size(13);
    /// ```
    pub fn new(data_range: f64) -> Self {
        assert!(data_range > 0.0, "data_range must be positive");
        Self {
            data_range,
            k1: 0.01,
            k2: 0.03,
            window_size: 11,
            sigma: 1.5,
        }
    }

    /// Sets a custom value for the k1 and k2 parameters of SSIM which are
    /// used for numerical stability.
    ///
    /// # Default values
    /// - k1: 0.01
    /// - k2: 0.03
    ///
    /// # Panics
    /// - If `k1` or `k2` is not positive.
    pub fn with_k1_k2(mut self, k1: f64, k2: f64) -> Self {
        assert!(k1 > 0.0, "k1 must be positive");
        assert!(k2 > 0.0, "k2 must be positive");
        self.k1 = k1;
        self.k2 = k2;
        self
    }

    /// Sets a custom window size for the Gaussian kernel used in SSIM. The
    /// window size must be a positive odd number.
    ///
    /// # Default value
    /// - window_size: 11
    ///
    /// # Panics
    /// - If `window_size` is not a positive odd number.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        assert!(
            window_size > 0 && window_size % 2 == 1,
            "window_size must be positive and an odd number"
        );
        self.window_size = window_size;
        self
    }

    /// Sets a custom sigma (standard deviation) for the Gaussian kernel used in SSIM.
    ///
    /// # Default value
    /// - sigma: 1.5
    ///
    /// # Panics
    /// - If `sigma` is not positive.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        self.sigma = sigma;
        self
    }
}

/// The SSIM (structural similarity index measure) metric for image quality assessment.
///
/// SSIM measures the perceived quality of images by comparing luminance,
/// contrast, and structure. Values range from -1 to 1, where 1 indicates
/// perfect structural similarity.
///
/// # Formula
/// ```text
/// SSIM(x, y) = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
/// ```
///
/// # Note
/// - This implementation uses separable Gaussian convolution for efficiency. Instead of a
///   single 2D convolution with a K by K kernel, it applies two 1D convolutions (horizontal
///   then vertical). This reduces the computational complexity from O(K^2) to O(2K) per pixel.
/// - SSIM is computed for each image first, and then it is averaged across all the images in the batch.
#[derive(Clone)]
pub struct SsimMetric<B: Backend> {
    name: MetricName,
    /// Internal state for numeric metric aggregation.
    state: NumericMetricState,
    /// Marker for backend type.
    _b: PhantomData<B>,
    /// Configuration for the metric.
    config: SsimMetricConfig,
}

impl<B: Backend> SsimMetric<B> {
    /// Creates a new SSIM metric with the given configuration.
    ///
    /// # Note
    /// The metric name format is "SSIM (dr={}, w={}, σ={})"
    /// where dr is the data range, w is the window size, sigma is the
    /// standard deviation. For example, the metric name might be
    /// "SSIM (dr=1.0, w=11, σ=1.5)".
    ///
    /// # Example
    /// ```ignore
    /// let ssim_config = SsimMetricConfig::new(1.0);
    /// let ssim_metric = SsimMetric::<B>::new(ssim_config);
    /// ```
    pub fn new(config: SsimMetricConfig) -> Self {
        Self {
            name: MetricName::new(format!(
                "SSIM (dr={}, w={}, σ={})",
                config.data_range, config.window_size, config.sigma,
            )),
            state: NumericMetricState::default(),
            config,
            _b: PhantomData,
        }
    }

    /// Overrides the default metric name which is "SSIM".
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = MetricName::new(name.to_string());
        self
    }

    /// Creates a 1D Gaussian kernel as a tensor.
    ///
    /// Returns a normalized kernel where all values sum to 1.
    /// The returned kernel will be reshaped by the `gaussian_conv_separable`
    /// associated function later.
    fn create_1d_gaussian_kernel(&self) -> Vec<f32> {
        let size = self.config.window_size;
        let sigma = self.config.sigma;
        let center = (size / 2) as f64;

        let mut kernel = vec![0.0f32; size];
        let mut sum = 0.0f64;

        for (i, v) in kernel.iter_mut().enumerate() {
            let x = i as f64 - center;
            let value = (-(x * x) / (2.0 * sigma * sigma)).exp();
            *v = value as f32;
            sum += value;
        }

        // Normalize so values sum to 1
        for v in kernel.iter_mut() {
            *v /= sum as f32;
        }

        kernel
    }

    /// Applies separable convolution using two 1D Gaussian kernels.
    ///
    /// # Arguments
    /// - `inputs`: Tensor of shape [N, C, H, W]
    /// - `kernel_1d`: The 1D Gaussian kernel values
    /// - `channels`: Number of channels for depthwise convolution.
    fn gaussian_conv_separable(
        &self,
        input: Tensor<B, 4>,
        kernel_1d: &[f32],
        channels: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let size = self.config.window_size;
        let padding = size / 2;

        // Create horizontal kernel: shape [C, 1, 1, K]
        let horizontal_kernel = Tensor::<B, 1>::from_floats(kernel_1d, device)
            .reshape([1, 1, 1, size]) // [1, 1, 1, K]
            .repeat_dim(0, channels); // [C, 1, 1, K]

        let vertical_kernel = Tensor::<B, 1>::from_floats(kernel_1d, device)
            .reshape([1, 1, size, 1]) // [1, 1, K, 1]
            .repeat_dim(0, channels); // [C, 1, K, 1]

        // Apply horizontal convolution
        let horizontal_conv_options = ConvOptions::new([1, 1], [0, padding], [1, 1], channels);
        let input_after_horizontal_conv =
            conv2d(input, horizontal_kernel, None, horizontal_conv_options);

        // Apply vertical convolution
        let vertical_conv_options = ConvOptions::new([1, 1], [padding, 0], [1, 1], channels);
        conv2d(
            input_after_horizontal_conv,
            vertical_kernel,
            None,
            vertical_conv_options,
        )
    }
}

impl<B: Backend> Metric for SsimMetric<B> {
    type Input = SsimInput<B>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let dims = item.outputs.dims();
        let batch_size = dims[0];
        let channels = dims[1];
        let device = item.outputs.device();

        let img_height = dims[2];
        let img_width = dims[3];
        assert!(
            img_height >= self.config.window_size && img_width >= self.config.window_size,
            "Image dimensions (H={}, W={}) must be >= window_size ({})",
            img_height,
            img_width,
            self.config.window_size
        );

        // Constants in SSIM formula used for numerical stability
        let c1 = (self.config.k1 * self.config.data_range).powi(2);
        let c2 = (self.config.k2 * self.config.data_range).powi(2);

        // Create 1D Gaussian kernel to apply separable convolutions twice (horizontally and vertically)
        let kernel_1d = self.create_1d_gaussian_kernel();

        // Compute mu_x and mu_y, their product and squares
        let x = item.outputs.clone();
        let y = item.targets.clone();
        let mu_x = self.gaussian_conv_separable(x.clone(), &kernel_1d, channels, &device);
        let mu_y = self.gaussian_conv_separable(y.clone(), &kernel_1d, channels, &device);
        let mu_x_mu_y = mu_x.clone() * mu_y.clone();
        let square_of_mu_x = mu_x.clone() * mu_x.clone();
        let square_of_mu_y = mu_y.clone() * mu_y.clone();

        // Compute var_x, var_y (which are the same as (sigma_x)^2 and (sigma_y)^2):
        // Var(X) = E[X^2] - E[X]^2
        // var_x = mu_of_x_squared - (mu_x * mu_x)
        let mu_of_x_squared =
            self.gaussian_conv_separable(x.clone() * x.clone(), &kernel_1d, channels, &device);
        let mu_of_y_squared =
            self.gaussian_conv_separable(y.clone() * y.clone(), &kernel_1d, channels, &device);
        let var_x = (mu_of_x_squared - square_of_mu_x.clone()).clamp_min(0.0);
        let var_y = (mu_of_y_squared - square_of_mu_y.clone()).clamp_min(0.0);

        // Compute the sample covariance of x and y: sigma_xy
        // Cov(X, Y) = E[XY] - E[X]E[Y]
        // sigma_xy = mu_xy - (mu_x * mu_y)
        let mu_xy = self.gaussian_conv_separable(x * y, &kernel_1d, channels, &device);
        let sigma_xy = mu_xy - mu_x_mu_y.clone();

        // Compute SSIM:
        // SSIM(x, y) = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
        let numerator = (mu_x_mu_y.mul_scalar(2.0) + c1) * (sigma_xy.mul_scalar(2.0) + c2);
        let denominator = (square_of_mu_x + square_of_mu_y + c1) * (var_x + var_y + c2);
        let ssim_tensor = numerator / denominator;

        // Average SSIM across all dimensions to get a single scalar value
        let ssim_per_image = ssim_tensor.mean_dims(&[1, 2, 3]);
        let avg_ssim = ssim_per_image.mean().into_scalar().elem::<f64>();

        self.state.update(
            avg_ssim,
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

impl<B: Backend> Numeric for SsimMetric<B> {
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
    use burn_core::tensor::{Distribution, Shape, TensorData};

    fn test_config() -> SsimMetricConfig {
        SsimMetricConfig::new(1.0)
            .with_window_size(3)
            .with_sigma(1.0)
    }

    #[test]
    fn test_ssim_perfect_similarity() {
        // When outputs exactly match targets, SSIM should be 1.0
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[
                [0.1_f32, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9],
            ]]]),
            &device,
        );
        let targets = outputs.clone();

        let mut metric = SsimMetric::<TestBackend>::new(test_config());
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        assert!(
            (ssim - 1.0).abs() < 0.001,
            "SSIM for identical images should be 1.0, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_completely_different() {
        // Constant black vs constant white
        // With constant images: SSIM = (2*mu_x*mu_y + C1) / (mu_x^2 + mu_y^2 + C1)
        // For x=0, y=1 with C1=(0.01)^2=0.0001: SSIM ≈ 0.0001 / (1 + 0.00001) = 0.00009999
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::zeros([1, 1, 4, 4], &device);
        let targets = Tensor::<TestBackend, 4>::ones([1, 1, 4, 4], &device);

        let mut metric = SsimMetric::<TestBackend>::new(test_config());
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        assert!(
            ssim < 0.0001,
            "SSIM for black vs white images should be very low, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_similar_images() {
        // Small perturbation should give high SSIM
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::full([1, 1, 4, 4], 0.5, &device);
        let targets = Tensor::<TestBackend, 4>::full([1, 1, 4, 4], 0.51, &device);

        let mut metric = SsimMetric::<TestBackend>::new(test_config());
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        assert!(
            ssim > 0.99,
            "SSIM for very similar images should be close to 1.0, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_batch_averaging() {
        // Batch of 2 images:
        // Image 1: identical (SSIM = 1.0)
        // Image 2: black vs white (SSIM ≈ 0)
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[
                    [0.5_f32, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                ]],
                [[
                    [0.0_f32, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]],
            ]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[
                    [0.5_f32, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                ]],
                [[
                    [1.0_f32, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]],
            ]),
            &device,
        );

        let mut metric = SsimMetric::<TestBackend>::new(test_config());
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        // Average of ~1.0 and ~0.0 should be around 0.5
        assert!(
            ssim > 0.49 && ssim < 0.51,
            "Average SSIM should be around 0.5, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_multichannel() {
        // Test with 3 channels (e.g., RGB)
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[
                [
                    [0.5_f32, 0.6, 0.7, 0.8],
                    [0.4, 0.5, 0.6, 0.7],
                    [0.3, 0.4, 0.5, 0.6],
                    [0.2, 0.3, 0.4, 0.5],
                ],
                [
                    [0.3_f32, 0.4, 0.5, 0.6],
                    [0.2, 0.3, 0.4, 0.5],
                    [0.1, 0.2, 0.3, 0.4],
                    [0.0, 0.1, 0.2, 0.3],
                ],
                [
                    [0.7_f32, 0.8, 0.9, 1.0],
                    [0.6, 0.7, 0.8, 0.9],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.4, 0.5, 0.6, 0.7],
                ],
            ]]),
            &device,
        );
        let targets = outputs.clone();

        let mut metric = SsimMetric::<TestBackend>::new(test_config());
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        assert!(
            (ssim - 1.0).abs() < 0.001,
            "SSIM for identical RGB images should be 1.0, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_symmetry() {
        // SSIM(x, y) should equal SSIM(y, x)
        // Symmetry is one of the mathematical properties of SSIM
        let device = Default::default();
        let img1 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[
                [0.1_f32, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9],
            ]]]),
            &device,
        );
        let img2 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[
                [0.2_f32, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9],
                [0.3, 0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9, 1.0],
            ]]]),
            &device,
        );

        let config = test_config();

        let mut metric1 = SsimMetric::<TestBackend>::new(config);
        let input1 = SsimInput::new(img1.clone(), img2.clone());
        let _entry = metric1.update(&input1, &MetricMetadata::fake());
        let ssim1 = metric1.value().current();

        let mut metric2 = SsimMetric::<TestBackend>::new(config);
        let input2 = SsimInput::new(img2, img1);
        let _entry = metric2.update(&input2, &MetricMetadata::fake());
        let ssim2 = metric2.value().current();

        assert!(
            (ssim1 - ssim2).abs() < 0.001,
            "SSIM should be symmetric: SSIM(x,y)={} vs SSIM(y,x)={}",
            ssim1,
            ssim2
        );
    }

    #[test]
    fn test_ssim_range() {
        // SSIM values should be in [-1, 1] range
        let device = Default::default();
        let shape = Shape::new([1, 1, 11, 11]);
        let distribution = Distribution::Uniform(0.0, 1.0);
        let outputs = Tensor::<TestBackend, 4>::random(shape.clone(), distribution, &device);
        let targets = Tensor::<TestBackend, 4>::random(shape, distribution, &device);

        let mut metric = SsimMetric::<TestBackend>::new(test_config());
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        assert!(
            ssim >= -1.0 && ssim <= 1.0,
            "SSIM should be in range [-1, 1], got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_running_average() {
        let device = Default::default();
        let mut metric = SsimMetric::<TestBackend>::new(test_config());

        // First update: identical images (SSIM = 1.0)
        let outputs1 = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[
                [0.5_f32, 0.6, 0.7, 0.8],
                [0.4, 0.5, 0.6, 0.7],
                [0.3, 0.4, 0.5, 0.6],
                [0.2, 0.3, 0.4, 0.5],
            ]]]),
            &device,
        );
        let targets1 = outputs1.clone();
        let input1 = SsimInput::new(outputs1, targets1);
        let _entry = metric.update(&input1, &MetricMetadata::fake());

        let ssim1 = metric.value().current();
        assert!(
            (ssim1 - 1.0).abs() < 0.001,
            "First update SSIM should be ~1.0, got {}",
            ssim1
        );

        // Second update: very different images (SSIM close to 0)
        let outputs2 = Tensor::<TestBackend, 4>::zeros([1, 1, 4, 4], &device);
        let targets2 = Tensor::<TestBackend, 4>::ones([1, 1, 4, 4], &device);
        let input2 = SsimInput::new(outputs2, targets2);
        let _entry = metric.update(&input2, &MetricMetadata::fake());

        // Running average should be around 0.5
        let running_avg = metric.running_value().current();
        assert!(
            running_avg > 0.49 && running_avg < 0.51,
            "Running average should be around 0.5, got {}",
            running_avg
        );
    }

    #[test]
    fn test_ssim_clear() {
        let device = Default::default();
        let mut metric = SsimMetric::<TestBackend>::new(test_config());

        let outputs = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[
                [0.5_f32, 0.6, 0.7, 0.8],
                [0.4, 0.5, 0.6, 0.7],
                [0.3, 0.4, 0.5, 0.6],
                [0.2, 0.3, 0.4, 0.5],
            ]]]),
            &device,
        );
        let targets = outputs.clone();
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        assert!(
            (ssim - 1.0).abs() < 0.001,
            "Expected SSIM ~1.0, got {}",
            ssim
        );

        // Clear and verify reset
        metric.clear();
        let ssim = metric.running_value().current();
        assert!(ssim.is_nan(), "Expected NaN after clear, got {}", ssim);
    }

    #[test]
    fn test_ssim_custom_name() {
        let config = SsimMetricConfig::new(1.0);
        let metric = SsimMetric::<TestBackend>::new(config).with_name("CustomSSIM");
        assert_eq!(metric.name().to_string(), "CustomSSIM");

        let metric = SsimMetric::<TestBackend>::new(test_config());
        assert_eq!(metric.name().to_string(), "SSIM (dr=1, w=3, σ=1)");

        let config = SsimMetricConfig::new(255.0);
        let metric = SsimMetric::<TestBackend>::new(config);
        assert_eq!(metric.name().to_string(), "SSIM (dr=255, w=11, σ=1.5)");
    }

    #[test]
    fn test_ssim_data_range_255() {
        // Test with 8-bit image range [0, 255]
        let device = Default::default();
        let shape = Shape::new([1, 1, 10, 10]);
        let distribution = Distribution::Uniform(0.0, 255.0);
        let outputs = Tensor::<TestBackend, 4>::random(shape.clone(), distribution, &device);
        let targets = outputs.clone();

        let config = SsimMetricConfig::new(255.0).with_window_size(3);
        let mut metric = SsimMetric::<TestBackend>::new(config);
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        assert!(
            (ssim - 1.0).abs() < 0.001,
            "SSIM for identical 8-bit images should be 1.0, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_large_batch() {
        let device = Default::default();
        let shape = Shape::new([20, 3, 30, 30]);
        let distribution = Distribution::Uniform(0.0, 1.0);
        let outputs = Tensor::<TestBackend, 4>::random(shape, distribution, &device);
        let targets = outputs.clone();

        let mut metric = SsimMetric::<TestBackend>::new(test_config());
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        assert!(
            (ssim - 1.0).abs() < 0.001,
            "SSIM for identical batch should be 1.0, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_default_window_size() {
        // Test with default window_size=11, need images >= 11x11
        let device = Default::default();
        let shape = Shape::new([1, 1, 1080, 1920]);
        let distribution = Distribution::Uniform(0.0, 1.0);
        let outputs = Tensor::<TestBackend, 4>::random(shape, distribution, &device);
        let targets = outputs.clone();

        let config = SsimMetricConfig::new(1.0); // default window_size=11
        let mut metric = SsimMetric::<TestBackend>::new(config);
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());

        let ssim = metric.value().current();
        assert!(
            (ssim - 1.0).abs() < 0.001,
            "SSIM with default window size should work and SSIM should be ~0.0, got {}",
            ssim
        );
    }

    #[test]
    fn test_ssim_attributes() {
        let config = SsimMetricConfig::new(1.0);
        let metric = SsimMetric::<TestBackend>::new(config);
        let attrs = metric.attributes();

        match attrs {
            MetricAttributes::Numeric(numeric_attrs) => {
                assert_eq!(numeric_attrs.unit, None);
                assert!(numeric_attrs.higher_is_better);
            }
            _ => panic!("Expected numeric attributes"),
        }
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_ssim_shape_mismatch() {
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::zeros([1, 1, 4, 4], &device);
        let targets = Tensor::<TestBackend, 4>::zeros([1, 1, 5, 5], &device);

        let _ = SsimInput::new(outputs, targets);
    }

    #[test]
    #[should_panic(expected = "Image dimensions (H=4, W=4) must be >= window_size (11)")]
    fn test_ssim_image_too_small() {
        let device = Default::default();
        let outputs = Tensor::<TestBackend, 4>::zeros([1, 1, 4, 4], &device);
        let targets = outputs.clone();

        // Default window_size=11, but image is only 4x4
        let config = SsimMetricConfig::new(1.0);
        let mut metric = SsimMetric::<TestBackend>::new(config);
        let input = SsimInput::new(outputs, targets);
        let _entry = metric.update(&input, &MetricMetadata::fake());
    }

    #[test]
    fn test_ssim_valid_k1_k2() {
        let config = SsimMetricConfig::new(1.0).with_k1_k2(0.015, 0.035);
        assert!(
            config.k1 == 0.015 && config.k2 == 0.035,
            "Expected k1=0.015 and k2=0.035, got k1={} and k2={}",
            config.k1,
            config.k2
        );
    }

    #[test]
    #[should_panic(expected = "data_range must be positive")]
    fn test_ssim_negative_data_range() {
        let _ = SsimMetricConfig::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "data_range must be positive")]
    fn test_ssim_zero_data_range() {
        let _ = SsimMetricConfig::new(0.0);
    }

    #[test]
    #[should_panic(expected = "k1 must be positive")]
    fn test_ssim_negative_k1() {
        let _ = SsimMetricConfig::new(1.0).with_k1_k2(-0.01, 0.03);
    }

    #[test]
    #[should_panic(expected = "k2 must be positive")]
    fn test_ssim_negative_k2() {
        let _ = SsimMetricConfig::new(1.0).with_k1_k2(0.01, -0.03);
    }

    #[test]
    #[should_panic(expected = "window_size must be positive and an odd number")]
    fn test_ssim_even_window_size() {
        let _ = SsimMetricConfig::new(1.0).with_window_size(10);
    }

    #[test]
    #[should_panic(expected = "window_size must be positive and an odd number")]
    fn test_ssim_zero_window_size() {
        let _ = SsimMetricConfig::new(1.0).with_window_size(0);
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn test_ssim_negative_sigma() {
        let _ = SsimMetricConfig::new(1.0).with_sigma(-1.5);
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn test_ssim_zero_sigma() {
        let _ = SsimMetricConfig::new(1.0).with_sigma(0.0);
    }
}

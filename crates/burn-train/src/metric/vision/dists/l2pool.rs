//! L2 Pooling layer for DISTS.
//!
//! L2 Pooling applies a Hanning window filter and computes the L2 norm
//! across the pooling window. This is used in DISTS instead of MaxPooling.

use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_nn::PaddingConfig2d;
use burn_nn::conv::{Conv2d, Conv2dConfig};

/// L2 Pooling layer configuration.
#[derive(Debug, Clone)]
pub struct L2Pool2dConfig {
    /// Kernel size for pooling
    pub kernel_size: usize,
    /// Stride for pooling
    pub stride: usize,
    /// Padding for pooling
    pub padding: usize,
}

impl Default for L2Pool2dConfig {
    fn default() -> Self {
        Self {
            kernel_size: 5,
            stride: 2,
            padding: 2,
        }
    }
}

impl L2Pool2dConfig {
    /// Create a new L2Pool2d configuration.
    #[allow(dead_code)]
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Initialize the L2Pool2d layer.
    pub fn init<B: Backend>(&self, channels: usize, device: &B::Device) -> L2Pool2d<B> {
        L2Pool2d::new(
            channels,
            self.kernel_size,
            self.stride,
            self.padding,
            device,
        )
    }
}

/// L2 Pooling layer.
///
/// Applies a 2D Hanning window filter followed by L2 normalization.
/// This provides smoother downsampling compared to MaxPooling.
#[derive(Module, Debug)]
pub struct L2Pool2d<B: Backend> {
    /// Depthwise convolution with Hanning kernel
    conv: Conv2d<B>,
}

impl<B: Backend> L2Pool2d<B> {
    /// Create a new L2Pool2d layer with Hanning window kernel.
    pub fn new(
        channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        device: &B::Device,
    ) -> Self {
        // Create Hanning kernel
        let kernel = Self::create_hanning_kernel(channels, kernel_size, device);

        // Create depthwise convolution (groups = channels)
        let mut conv = Conv2dConfig::new([channels, channels], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(
                padding, padding, padding, padding,
            ))
            .with_groups(channels)
            .with_bias(false)
            .init(device);

        // Set the kernel weights to Hanning window
        conv.weight = burn::module::Param::from_tensor(kernel);

        Self { conv }
    }

    /// Create a Hanning kernel for depthwise convolution.
    /// Output shape: [channels, 1, kernel_size, kernel_size]
    fn create_hanning_kernel<B2: Backend>(
        channels: usize,
        kernel_size: usize,
        device: &B2::Device,
    ) -> Tensor<B2, 4> {
        // Create 1D Hanning window
        let mut hanning_1d = Vec::with_capacity(kernel_size);
        for i in 0..kernel_size {
            let n = i as f32;
            let n_minus_1 = (kernel_size - 1) as f32;
            let value = if n_minus_1 == 0.0 {
                1.0
            } else {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * n / n_minus_1).cos())
            };
            hanning_1d.push(value);
        }

        // Create 2D Hanning window by outer product
        let mut hanning_2d = Vec::with_capacity(kernel_size * kernel_size);
        let mut sum = 0.0;
        for i in 0..kernel_size {
            for j in 0..kernel_size {
                let value = hanning_1d[i] * hanning_1d[j];
                hanning_2d.push(value);
                sum += value;
            }
        }

        // Normalize
        for v in hanning_2d.iter_mut() {
            *v /= sum;
        }

        // Create tensor of shape [1, 1, kernel_size, kernel_size]
        let kernel_single = Tensor::<B2, 1>::from_floats(hanning_2d.as_slice(), device).reshape([
            1,
            1,
            kernel_size,
            kernel_size,
        ]);

        // Expand to [channels, 1, kernel_size, kernel_size]
        kernel_single.repeat_dim(0, channels)
    }

    /// Apply L2 pooling to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, channels, height, width]`
    ///
    /// # Returns
    ///
    /// Pooled tensor with reduced spatial dimensions.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Square the input
        let x_sq = x.clone().mul(x);

        // Apply depthwise convolution with Hanning kernel
        let pooled = self.conv.forward(x_sq);

        // Take square root for L2 norm
        // Add small epsilon to avoid sqrt of negative numbers due to numerical errors
        pooled.clamp_min(1e-10).sqrt()
    }
}

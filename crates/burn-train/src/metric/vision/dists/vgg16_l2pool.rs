//! VGG16 feature extractor with L2 Pooling for DISTS.
//!
//! This module implements the VGG16 backbone used in DISTS,
//! with L2Pooling replacing MaxPooling for smoother feature extraction.

use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn_nn::PaddingConfig2d;
use burn_nn::conv::{Conv2d, Conv2dConfig};

use super::l2pool::{L2Pool2d, L2Pool2dConfig};

/// VGG16 feature extractor with L2 Pooling for DISTS.
///
/// Extracts features from 5 stages of VGG16, using L2Pooling
/// instead of MaxPooling for smoother downsampling.
///
/// Output channels per stage: [64, 128, 256, 512, 512]
#[derive(Module, Debug)]
pub struct Vgg16L2PoolExtractor<B: Backend> {
    // Stage 1: 2 conv layers, 64 channels
    pub(crate) conv1_1: Conv2d<B>,
    pub(crate) conv1_2: Conv2d<B>,
    pub(crate) pool1: L2Pool2d<B>,

    // Stage 2: 2 conv layers, 128 channels
    pub(crate) conv2_1: Conv2d<B>,
    pub(crate) conv2_2: Conv2d<B>,
    pub(crate) pool2: L2Pool2d<B>,

    // Stage 3: 3 conv layers, 256 channels
    pub(crate) conv3_1: Conv2d<B>,
    pub(crate) conv3_2: Conv2d<B>,
    pub(crate) conv3_3: Conv2d<B>,
    pub(crate) pool3: L2Pool2d<B>,

    // Stage 4: 3 conv layers, 512 channels
    pub(crate) conv4_1: Conv2d<B>,
    pub(crate) conv4_2: Conv2d<B>,
    pub(crate) conv4_3: Conv2d<B>,
    pub(crate) pool4: L2Pool2d<B>,

    // Stage 5: 3 conv layers, 512 channels
    pub(crate) conv5_1: Conv2d<B>,
    pub(crate) conv5_2: Conv2d<B>,
    pub(crate) conv5_3: Conv2d<B>,
}

impl<B: Backend> Vgg16L2PoolExtractor<B> {
    /// Create a new VGG16 feature extractor with L2 Pooling.
    pub fn new(device: &B::Device) -> Self {
        let pool_config = L2Pool2dConfig::default();

        Self {
            // Stage 1
            conv1_1: Conv2dConfig::new([3, 64], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv1_2: Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool1: pool_config.init(64, device),

            // Stage 2
            conv2_1: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2_2: Conv2dConfig::new([128, 128], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool2: pool_config.init(128, device),

            // Stage 3
            conv3_1: Conv2dConfig::new([128, 256], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv3_2: Conv2dConfig::new([256, 256], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv3_3: Conv2dConfig::new([256, 256], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool3: pool_config.init(256, device),

            // Stage 4
            conv4_1: Conv2dConfig::new([256, 512], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv4_2: Conv2dConfig::new([512, 512], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv4_3: Conv2dConfig::new([512, 512], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool4: pool_config.init(512, device),

            // Stage 5
            conv5_1: Conv2dConfig::new([512, 512], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv5_2: Conv2dConfig::new([512, 512], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv5_3: Conv2dConfig::new([512, 512], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
        }
    }

    /// Extract features from all 5 stages.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, 3, height, width]`
    ///
    /// # Returns
    ///
    /// Vector of 6 feature tensors:
    /// - Stage 0: Input image [batch, 3, H, W]
    /// - Stage 1: After conv1 [batch, 64, H/2, W/2]
    /// - Stage 2: After conv2 [batch, 128, H/4, W/4]
    /// - Stage 3: After conv3 [batch, 256, H/8, W/8]
    /// - Stage 4: After conv4 [batch, 512, H/16, W/16]
    /// - Stage 5: After conv5 [batch, 512, H/32, W/32]
    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let mut features = Vec::with_capacity(6);

        // Stage 0: Input image
        features.push(x.clone());

        // Stage 1
        let x = relu(self.conv1_1.forward(x));
        let x = relu(self.conv1_2.forward(x));
        features.push(x.clone());
        let x = self.pool1.forward(x);

        // Stage 2
        let x = relu(self.conv2_1.forward(x));
        let x = relu(self.conv2_2.forward(x));
        features.push(x.clone());
        let x = self.pool2.forward(x);

        // Stage 3
        let x = relu(self.conv3_1.forward(x));
        let x = relu(self.conv3_2.forward(x));
        let x = relu(self.conv3_3.forward(x));
        features.push(x.clone());
        let x = self.pool3.forward(x);

        // Stage 4
        let x = relu(self.conv4_1.forward(x));
        let x = relu(self.conv4_2.forward(x));
        let x = relu(self.conv4_3.forward(x));
        features.push(x.clone());
        let x = self.pool4.forward(x);

        // Stage 5
        let x = relu(self.conv5_1.forward(x));
        let x = relu(self.conv5_2.forward(x));
        let x = relu(self.conv5_3.forward(x));
        features.push(x);

        features
    }
}

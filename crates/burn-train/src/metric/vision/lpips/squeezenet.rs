//! SqueezeNet feature extractor for LPIPS.

use burn_core as burn;

use burn::module::Module;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_nn::conv::{Conv2d, Conv2dConfig};
use burn_nn::PaddingConfig2d;

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
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
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
        Tensor::cat(vec![e1, e3], 1)
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
            fire1: FireModule::new(64, 16, 64, 64, device),    // -> 128
            fire2: FireModule::new(128, 16, 64, 64, device),   // -> 128
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

/// 3x3 max pooling with stride 2, ceil mode (for SqueezeNet).
fn max_pool2d_squeeze<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    burn_core::tensor::module::max_pool2d(x, [3, 3], [2, 2], [0, 0], [1, 1], true)
}

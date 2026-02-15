//! AlexNet feature extractor for LPIPS.

use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn_nn::PaddingConfig2d;
use burn_nn::conv::{Conv2d, Conv2dConfig};

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
                .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
                .with_bias(true)
                .init(device),
            // Conv2: 64 -> 192, 5x5, stride 1, padding 2
            conv2: Conv2dConfig::new([64, 192], [5, 5])
                .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
                .with_bias(true)
                .init(device),
            // Conv3: 192 -> 384, 3x3, stride 1, padding 1
            conv3: Conv2dConfig::new([192, 384], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(true)
                .init(device),
            // Conv4: 384 -> 256, 3x3, stride 1, padding 1
            conv4: Conv2dConfig::new([384, 256], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(true)
                .init(device),
            // Conv5: 256 -> 256, 3x3, stride 1, padding 1
            conv5: Conv2dConfig::new([256, 256], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
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

/// 3x3 max pooling with stride 2 (for AlexNet).
fn max_pool2d_alex<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    burn_core::tensor::module::max_pool2d(x, [3, 3], [2, 2], [0, 0], [1, 1], false)
}

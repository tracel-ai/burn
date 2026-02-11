//! VGG16 feature extractor for LPIPS.

use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn_nn::PaddingConfig2d;
use burn_nn::conv::{Conv2d, Conv2dConfig};

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

/// 2x2 max pooling with stride 2.
fn max_pool2d<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    burn_core::tensor::module::max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1], false)
}

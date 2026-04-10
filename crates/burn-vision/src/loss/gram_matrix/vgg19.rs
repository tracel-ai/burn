use burn_core as burn;

use crate::PaddingConfig2d;
use crate::conv::{Conv2d, Conv2dConfig};
use burn::module::Module;
use burn::tensor::{
    Device, Tensor,
    activation::relu,
    module::{avg_pool2d, max_pool2d},
};

/// VGG19 feature extractor for the Gram Matrix Loss.
///
/// This module implements the VGG19 architecture up to the 5th convolutional block.
/// It is specifically tailored for Neural Style Transfer and Gram Matrix Loss,
/// extracting and flattening features from the following 5 layers:
/// - `conv1_1`
/// - `conv2_1`
/// - `conv3_1`
/// - `conv4_1`
/// - `conv5_1`
#[derive(Module, Debug)]
pub struct Vgg19 {
    use_avg_pool: bool,

    // Block 1
    // Field is made public for testing whether the weights are frozen or not
    pub conv1_1: Conv2d,
    conv1_2: Conv2d,

    // Block 2
    conv2_1: Conv2d,
    conv2_2: Conv2d,

    // Block 3
    conv3_1: Conv2d,
    conv3_2: Conv2d,
    conv3_3: Conv2d,
    conv3_4: Conv2d,

    // Block 4
    conv4_1: Conv2d,
    conv4_2: Conv2d,
    conv4_3: Conv2d,
    conv4_4: Conv2d,

    // Block 5
    conv5_1: Conv2d,
}

impl Vgg19 {
    /// Creates a new VGG19 feature extractor.
    ///
    /// The network is initialized with standard VGG19 configurations (3x3 kernels,
    /// stride 1, padding 1). Note that the weights are randomly initialized here so
    /// they should be overwritten by `load_vgg19_weights` before use.
    pub fn new(use_avg_pool: bool, device: &Device) -> Self {
        // All convolutions use a kernel size of 3 by 3, stride of 1, and
        // padding of 1.
        // This combination of kernel size and padding preserves input
        // dimensions. Thus, `PaddingConfig2d::Same` can be used instead.
        let conv_config = |in_ch, out_ch| {
            Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };

        Self {
            use_avg_pool,
            // Block 1
            conv1_1: conv_config(3, 64),
            conv1_2: conv_config(64, 64),
            // Block 2
            conv2_1: conv_config(64, 128),
            conv2_2: conv_config(128, 128),
            // Block 3
            conv3_1: conv_config(128, 256),
            conv3_2: conv_config(256, 256),
            conv3_3: conv_config(256, 256),
            conv3_4: conv_config(256, 256),
            // Block 4
            conv4_1: conv_config(256, 512),
            conv4_2: conv_config(512, 512),
            conv4_3: conv_config(512, 512),
            conv4_4: conv_config(512, 512),
            // Block 5
            conv5_1: conv_config(512, 512),
        }
    }

    /// Performs a forward pass to extract features for the Gram Matrix Loss.
    ///
    /// # Arguments
    ///
    /// - `x` - Input image tensor of shape `[batch_size, 3, height, width]`.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `features`: A `Vec` of 5 tensors, each representing the flattened feature map
    ///    from one of the target layers. Shape of each tensor: `[batch_size, channels, height * width]`.
    /// - `normalization_factors`: A `Vec` of 5 `f32` values, representing the normalization
    ///    factor `4 * N^2 * M^2` for each layer, used to scale the Gram matrix loss.
    pub fn forward(&self, x: Tensor<4>) -> Vec<Tensor<3>> {
        let pool_2d = |x| {
            if self.use_avg_pool {
                avg_pool2d(x, [2, 2], [2, 2], [0, 0], false, false)
            } else {
                max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1], false)
            }
        };

        let mut features = Vec::with_capacity(5);

        // Block 1
        let x1_1 = relu(self.conv1_1.forward(x));
        let flattened_x1_1 = x1_1.clone().flatten(2, 3);
        features.push(flattened_x1_1);
        let x1_2 = relu(self.conv1_2.forward(x1_1));
        let x1 = pool_2d(x1_2);

        // Block 2
        let x2_1 = relu(self.conv2_1.forward(x1));
        let flattened_x2_1 = x2_1.clone().flatten(2, 3);
        features.push(flattened_x2_1);
        let x2_2 = relu(self.conv2_2.forward(x2_1));
        let x2 = pool_2d(x2_2);

        // Block 3
        let x3_1 = relu(self.conv3_1.forward(x2));
        let flattened_x3_1 = x3_1.clone().flatten(2, 3);
        features.push(flattened_x3_1);
        let x3_2 = relu(self.conv3_2.forward(x3_1));
        let x3_3 = relu(self.conv3_3.forward(x3_2));
        let x3_4 = relu(self.conv3_4.forward(x3_3));
        let x3 = pool_2d(x3_4);

        // Block 4
        let x4_1 = relu(self.conv4_1.forward(x3));
        let flattened_x4_1 = x4_1.clone().flatten(2, 3);
        features.push(flattened_x4_1);
        let x4_2 = relu(self.conv4_2.forward(x4_1));
        let x4_3 = relu(self.conv4_3.forward(x4_2));
        let x4_4 = relu(self.conv4_4.forward(x4_3));
        let x4 = pool_2d(x4_4);

        // Block 5
        let x5_1 = relu(self.conv5_1.forward(x4));
        let flattened_x5_1 = x5_1.flatten(2, 3);
        features.push(flattened_x5_1);

        features
    }
}

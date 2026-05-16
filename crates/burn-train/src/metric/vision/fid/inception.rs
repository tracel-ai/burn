//! InceptionV3 feature extractor for FID (pytorch-fid variant with TF-ported weights).
//!
//! Reference: <https://github.com/mseitzer/pytorch-fid>

use burn_core as burn;

use burn::module::Module;
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn_nn::conv::{Conv2d, Conv2dConfig};
use burn_nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};

/// Conv2d + BatchNorm + ReLU building block.
#[derive(Module, Debug)]
pub struct BasicConv2d {
    conv: Conv2d,
    bn: BatchNorm,
}

impl BasicConv2d {
    pub fn new(conv_config: Conv2dConfig, device: &Device) -> Self {
        let out_channels = conv_config.channels[1];
        Self {
            conv: conv_config.with_bias(false).init(device),
            bn: BatchNormConfig::new(out_channels)
                .with_epsilon(0.001)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        relu(self.bn.forward(self.conv.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct InceptionA {
    branch1x1: BasicConv2d,
    branch5x5_1: BasicConv2d,
    branch5x5_2: BasicConv2d,
    branch3x3dbl_1: BasicConv2d,
    branch3x3dbl_2: BasicConv2d,
    branch3x3dbl_3: BasicConv2d,
    branch_pool: BasicConv2d,
}

impl InceptionA {
    pub fn new(in_channels: usize, pool_features: usize, device: &Device) -> Self {
        Self {
            branch1x1: BasicConv2d::new(Conv2dConfig::new([in_channels, 64], [1, 1]), device),
            branch5x5_1: BasicConv2d::new(Conv2dConfig::new([in_channels, 48], [1, 1]), device),
            branch5x5_2: BasicConv2d::new(
                Conv2dConfig::new([48, 64], [5, 5])
                    .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2)),
                device,
            ),
            branch3x3dbl_1: BasicConv2d::new(Conv2dConfig::new([in_channels, 64], [1, 1]), device),
            branch3x3dbl_2: BasicConv2d::new(
                Conv2dConfig::new([64, 96], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
                device,
            ),
            branch3x3dbl_3: BasicConv2d::new(
                Conv2dConfig::new([96, 96], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
                device,
            ),
            branch_pool: BasicConv2d::new(
                Conv2dConfig::new([in_channels, pool_features], [1, 1]),
                device,
            ),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let branch1x1 = self.branch1x1.forward(x.clone());

        let branch5x5 = self.branch5x5_1.forward(x.clone());
        let branch5x5 = self.branch5x5_2.forward(branch5x5);

        let branch3x3dbl = self.branch3x3dbl_1.forward(x.clone());
        let branch3x3dbl = self.branch3x3dbl_2.forward(branch3x3dbl);
        let branch3x3dbl = self.branch3x3dbl_3.forward(branch3x3dbl);

        let branch_pool =
            burn_core::tensor::module::avg_pool2d(x, [3, 3], [1, 1], [1, 1], false, false);
        let branch_pool = self.branch_pool.forward(branch_pool);

        Tensor::cat(vec![branch1x1, branch5x5, branch3x3dbl, branch_pool], 1)
    }
}

#[derive(Module, Debug)]
pub struct InceptionB {
    branch3x3: BasicConv2d,
    branch3x3dbl_1: BasicConv2d,
    branch3x3dbl_2: BasicConv2d,
    branch3x3dbl_3: BasicConv2d,
}

impl InceptionB {
    pub fn new(in_channels: usize, device: &Device) -> Self {
        Self {
            branch3x3: BasicConv2d::new(
                Conv2dConfig::new([in_channels, 384], [3, 3]).with_stride([2, 2]),
                device,
            ),
            branch3x3dbl_1: BasicConv2d::new(Conv2dConfig::new([in_channels, 64], [1, 1]), device),
            branch3x3dbl_2: BasicConv2d::new(
                Conv2dConfig::new([64, 96], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
                device,
            ),
            branch3x3dbl_3: BasicConv2d::new(
                Conv2dConfig::new([96, 96], [3, 3]).with_stride([2, 2]),
                device,
            ),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let branch3x3 = self.branch3x3.forward(x.clone());

        let branch3x3dbl = self.branch3x3dbl_1.forward(x.clone());
        let branch3x3dbl = self.branch3x3dbl_2.forward(branch3x3dbl);
        let branch3x3dbl = self.branch3x3dbl_3.forward(branch3x3dbl);

        let branch_pool =
            burn_core::tensor::module::max_pool2d(x, [3, 3], [2, 2], [0, 0], [1, 1], false);

        Tensor::cat(vec![branch3x3, branch3x3dbl, branch_pool], 1)
    }
}

#[derive(Module, Debug)]
pub struct InceptionC {
    branch1x1: BasicConv2d,
    branch7x7_1: BasicConv2d,
    branch7x7_2: BasicConv2d,
    branch7x7_3: BasicConv2d,
    branch7x7dbl_1: BasicConv2d,
    branch7x7dbl_2: BasicConv2d,
    branch7x7dbl_3: BasicConv2d,
    branch7x7dbl_4: BasicConv2d,
    branch7x7dbl_5: BasicConv2d,
    branch_pool: BasicConv2d,
}

impl InceptionC {
    pub fn new(in_channels: usize, channels_7x7: usize, device: &Device) -> Self {
        let c7 = channels_7x7;
        Self {
            branch1x1: BasicConv2d::new(Conv2dConfig::new([in_channels, 192], [1, 1]), device),
            branch7x7_1: BasicConv2d::new(Conv2dConfig::new([in_channels, c7], [1, 1]), device),
            branch7x7_2: BasicConv2d::new(
                Conv2dConfig::new([c7, c7], [1, 7])
                    .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3)),
                device,
            ),
            branch7x7_3: BasicConv2d::new(
                Conv2dConfig::new([c7, 192], [7, 1])
                    .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0)),
                device,
            ),
            branch7x7dbl_1: BasicConv2d::new(Conv2dConfig::new([in_channels, c7], [1, 1]), device),
            branch7x7dbl_2: BasicConv2d::new(
                Conv2dConfig::new([c7, c7], [7, 1])
                    .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0)),
                device,
            ),
            branch7x7dbl_3: BasicConv2d::new(
                Conv2dConfig::new([c7, c7], [1, 7])
                    .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3)),
                device,
            ),
            branch7x7dbl_4: BasicConv2d::new(
                Conv2dConfig::new([c7, c7], [7, 1])
                    .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0)),
                device,
            ),
            branch7x7dbl_5: BasicConv2d::new(
                Conv2dConfig::new([c7, 192], [1, 7])
                    .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3)),
                device,
            ),
            branch_pool: BasicConv2d::new(Conv2dConfig::new([in_channels, 192], [1, 1]), device),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let branch1x1 = self.branch1x1.forward(x.clone());

        let branch7x7 = self.branch7x7_1.forward(x.clone());
        let branch7x7 = self.branch7x7_2.forward(branch7x7);
        let branch7x7 = self.branch7x7_3.forward(branch7x7);

        let branch7x7dbl = self.branch7x7dbl_1.forward(x.clone());
        let branch7x7dbl = self.branch7x7dbl_2.forward(branch7x7dbl);
        let branch7x7dbl = self.branch7x7dbl_3.forward(branch7x7dbl);
        let branch7x7dbl = self.branch7x7dbl_4.forward(branch7x7dbl);
        let branch7x7dbl = self.branch7x7dbl_5.forward(branch7x7dbl);

        let branch_pool =
            burn_core::tensor::module::avg_pool2d(x, [3, 3], [1, 1], [1, 1], false, false);
        let branch_pool = self.branch_pool.forward(branch_pool);

        Tensor::cat(vec![branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)
    }
}

#[derive(Module, Debug)]
pub struct InceptionD {
    branch3x3_1: BasicConv2d,
    branch3x3_2: BasicConv2d,
    branch7x7x3_1: BasicConv2d,
    branch7x7x3_2: BasicConv2d,
    branch7x7x3_3: BasicConv2d,
    branch7x7x3_4: BasicConv2d,
}

impl InceptionD {
    pub fn new(in_channels: usize, device: &Device) -> Self {
        Self {
            branch3x3_1: BasicConv2d::new(Conv2dConfig::new([in_channels, 192], [1, 1]), device),
            branch3x3_2: BasicConv2d::new(
                Conv2dConfig::new([192, 320], [3, 3]).with_stride([2, 2]),
                device,
            ),
            branch7x7x3_1: BasicConv2d::new(Conv2dConfig::new([in_channels, 192], [1, 1]), device),
            branch7x7x3_2: BasicConv2d::new(
                Conv2dConfig::new([192, 192], [1, 7])
                    .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3)),
                device,
            ),
            branch7x7x3_3: BasicConv2d::new(
                Conv2dConfig::new([192, 192], [7, 1])
                    .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0)),
                device,
            ),
            branch7x7x3_4: BasicConv2d::new(
                Conv2dConfig::new([192, 192], [3, 3]).with_stride([2, 2]),
                device,
            ),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let branch3x3 = self.branch3x3_1.forward(x.clone());
        let branch3x3 = self.branch3x3_2.forward(branch3x3);

        let branch7x7x3 = self.branch7x7x3_1.forward(x.clone());
        let branch7x7x3 = self.branch7x7x3_2.forward(branch7x7x3);
        let branch7x7x3 = self.branch7x7x3_3.forward(branch7x7x3);
        let branch7x7x3 = self.branch7x7x3_4.forward(branch7x7x3);

        let branch_pool =
            burn_core::tensor::module::max_pool2d(x, [3, 3], [2, 2], [0, 0], [1, 1], false);

        Tensor::cat(vec![branch3x3, branch7x7x3, branch_pool], 1)
    }
}

#[derive(Module, Debug)]
pub struct InceptionE {
    branch1x1: BasicConv2d,
    branch3x3_1: BasicConv2d,
    branch3x3_2a: BasicConv2d,
    branch3x3_2b: BasicConv2d,
    branch3x3dbl_1: BasicConv2d,
    branch3x3dbl_2: BasicConv2d,
    branch3x3dbl_3a: BasicConv2d,
    branch3x3dbl_3b: BasicConv2d,
    branch_pool: BasicConv2d,
    #[module(skip)]
    use_max_pool: bool,
}

impl InceptionE {
    pub fn new(in_channels: usize, use_max_pool: bool, device: &Device) -> Self {
        Self {
            branch1x1: BasicConv2d::new(Conv2dConfig::new([in_channels, 320], [1, 1]), device),
            branch3x3_1: BasicConv2d::new(Conv2dConfig::new([in_channels, 384], [1, 1]), device),
            branch3x3_2a: BasicConv2d::new(
                Conv2dConfig::new([384, 384], [1, 3])
                    .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1)),
                device,
            ),
            branch3x3_2b: BasicConv2d::new(
                Conv2dConfig::new([384, 384], [3, 1])
                    .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0)),
                device,
            ),
            branch3x3dbl_1: BasicConv2d::new(Conv2dConfig::new([in_channels, 448], [1, 1]), device),
            branch3x3dbl_2: BasicConv2d::new(
                Conv2dConfig::new([448, 384], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
                device,
            ),
            branch3x3dbl_3a: BasicConv2d::new(
                Conv2dConfig::new([384, 384], [1, 3])
                    .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1)),
                device,
            ),
            branch3x3dbl_3b: BasicConv2d::new(
                Conv2dConfig::new([384, 384], [3, 1])
                    .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0)),
                device,
            ),
            branch_pool: BasicConv2d::new(Conv2dConfig::new([in_channels, 192], [1, 1]), device),
            use_max_pool,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let branch1x1 = self.branch1x1.forward(x.clone());

        let branch3x3 = self.branch3x3_1.forward(x.clone());
        let branch3x3_a = self.branch3x3_2a.forward(branch3x3.clone());
        let branch3x3_b = self.branch3x3_2b.forward(branch3x3);
        let branch3x3 = Tensor::cat(vec![branch3x3_a, branch3x3_b], 1);

        let branch3x3dbl = self.branch3x3dbl_1.forward(x.clone());
        let branch3x3dbl = self.branch3x3dbl_2.forward(branch3x3dbl);
        let branch3x3dbl_a = self.branch3x3dbl_3a.forward(branch3x3dbl.clone());
        let branch3x3dbl_b = self.branch3x3dbl_3b.forward(branch3x3dbl);
        let branch3x3dbl = Tensor::cat(vec![branch3x3dbl_a, branch3x3dbl_b], 1);

        let branch_pool = if self.use_max_pool {
            burn_core::tensor::module::max_pool2d(x, [3, 3], [1, 1], [1, 1], [1, 1], false)
        } else {
            burn_core::tensor::module::avg_pool2d(x, [3, 3], [1, 1], [1, 1], false, false)
        };
        let branch_pool = self.branch_pool.forward(branch_pool);

        Tensor::cat(vec![branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)
    }
}

/// InceptionV3 feature extractor for FID computation.
///
/// Outputs a 2048-dimensional feature vector per image, matching the
/// pytorch-fid variant (TF-ported weights).
#[derive(Module, Debug)]
pub struct InceptionV3FeatureExtractor {
    // Stem
    conv2d_1a: BasicConv2d,
    conv2d_2a: BasicConv2d,
    conv2d_2b: BasicConv2d,
    conv2d_3b: BasicConv2d,
    conv2d_4a: BasicConv2d,
    // Inception blocks
    mixed_5b: InceptionA,
    mixed_5c: InceptionA,
    mixed_5d: InceptionA,
    mixed_6a: InceptionB,
    mixed_6b: InceptionC,
    mixed_6c: InceptionC,
    mixed_6d: InceptionC,
    mixed_6e: InceptionC,
    mixed_7a: InceptionD,
    mixed_7b: InceptionE,
    mixed_7c: InceptionE,
}

impl InceptionV3FeatureExtractor {
    /// Creates a new feature extractor with random weights.
    pub fn new(device: &Device) -> Self {
        Self {
            // Stem: 3 -> 32 -> 32 -> 64 -> 80 -> 192
            conv2d_1a: BasicConv2d::new(
                Conv2dConfig::new([3, 32], [3, 3]).with_stride([2, 2]),
                device,
            ),
            conv2d_2a: BasicConv2d::new(Conv2dConfig::new([32, 32], [3, 3]), device),
            conv2d_2b: BasicConv2d::new(
                Conv2dConfig::new([32, 64], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
                device,
            ),
            conv2d_3b: BasicConv2d::new(Conv2dConfig::new([64, 80], [1, 1]), device),
            conv2d_4a: BasicConv2d::new(Conv2dConfig::new([80, 192], [3, 3]), device),
            mixed_5b: InceptionA::new(192, 32, device),
            mixed_5c: InceptionA::new(256, 64, device),
            mixed_5d: InceptionA::new(288, 64, device),
            mixed_6a: InceptionB::new(288, device),
            mixed_6b: InceptionC::new(768, 128, device),
            mixed_6c: InceptionC::new(768, 160, device),
            mixed_6d: InceptionC::new(768, 160, device),
            mixed_6e: InceptionC::new(768, 192, device),
            mixed_7a: InceptionD::new(768, device),
            mixed_7b: InceptionE::new(1280, false, device),
            mixed_7c: InceptionE::new(2048, true, device),
        }
    }

    /// Extract 2048-dim features. Input is resized to 299x299 via bilinear
    /// interpolation to match the pytorch-fid reference.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let [batch, _, h, w] = x.dims();

        let x = if h != 299 || w != 299 {
            burn_core::tensor::module::interpolate(
                x,
                [299, 299],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            )
        } else {
            x
        };

        // Stem
        let x = self.conv2d_1a.forward(x);
        let x = self.conv2d_2a.forward(x);
        let x = self.conv2d_2b.forward(x);
        let x = burn_core::tensor::module::max_pool2d(x, [3, 3], [2, 2], [0, 0], [1, 1], false);
        let x = self.conv2d_3b.forward(x);
        let x = self.conv2d_4a.forward(x);
        let x = burn_core::tensor::module::max_pool2d(x, [3, 3], [2, 2], [0, 0], [1, 1], false);

        // InceptionA
        let x = self.mixed_5b.forward(x);
        let x = self.mixed_5c.forward(x);
        let x = self.mixed_5d.forward(x);

        // InceptionB (reduction)
        let x = self.mixed_6a.forward(x);

        // InceptionC
        let x = self.mixed_6b.forward(x);
        let x = self.mixed_6c.forward(x);
        let x = self.mixed_6d.forward(x);
        let x = self.mixed_6e.forward(x);

        // InceptionD (reduction)
        let x = self.mixed_7a.forward(x);

        // InceptionE
        let x = self.mixed_7b.forward(x);
        let x = self.mixed_7c.forward(x);

        // Global average pool -> [N, 2048]
        let x = burn_core::tensor::module::adaptive_avg_pool2d(x, [1, 1]);
        x.reshape([batch, 2048])
    }
}

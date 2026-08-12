//! Minimal ResNet-18 inference fixture.
//!
//! Adapted from `tracel-ai/models/resnet-burn`, itself derived from
//! `torchvision.models.resnet`. See `NOTICE.md` in this directory.

use core::f64::consts::SQRT_2;

use burn_core as burn;
use burn_core::module::Module;
use burn_nn::{
    BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PaddingConfig2d, Relu,
    conv::{Conv2d, Conv2dConfig},
    pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
};
use burn_tensor::{Device, Tensor};

#[derive(Module, Debug)]
struct Downsample {
    conv: Conv2d,
    bn: BatchNorm,
}

impl Downsample {
    fn new(in_channels: usize, out_channels: usize, stride: usize, device: &Device) -> Self {
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };
        Self {
            conv: Conv2dConfig::new([in_channels, out_channels], [1, 1])
                .with_stride([stride, stride])
                .with_bias(false)
                .with_initializer(initializer)
                .init(device),
            bn: BatchNormConfig::new(out_channels).init(device),
        }
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.bn.forward(self.conv.forward(input))
    }
}

#[derive(Module, Debug)]
struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    relu: Relu,
    conv2: Conv2d,
    bn2: BatchNorm,
    downsample: Option<Downsample>,
}

impl BasicBlock {
    fn new(in_channels: usize, out_channels: usize, stride: usize, device: &Device) -> Self {
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };
        let conv = |channels, stride, initializer| {
            Conv2dConfig::new(channels, [3, 3])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .with_initializer(initializer)
                .init(device)
        };
        Self {
            conv1: conv([in_channels, out_channels], stride, initializer.clone()),
            bn1: BatchNormConfig::new(out_channels).init(device),
            relu: Relu::new(),
            conv2: conv([out_channels, out_channels], 1, initializer),
            bn2: BatchNormConfig::new(out_channels).init(device),
            downsample: (in_channels != out_channels || stride != 1)
                .then(|| Downsample::new(in_channels, out_channels, stride, device)),
        }
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let identity = input.clone();
        let out = self
            .relu
            .forward(self.bn1.forward(self.conv1.forward(input)));
        let out = self.bn2.forward(self.conv2.forward(out));
        let identity = match &self.downsample {
            Some(downsample) => downsample.forward(identity),
            None => identity,
        };
        self.relu.forward(out + identity)
    }
}

#[derive(Module, Debug)]
struct LayerBlock {
    blocks: Vec<BasicBlock>,
}

impl LayerBlock {
    fn new(in_channels: usize, out_channels: usize, stride: usize, device: &Device) -> Self {
        Self {
            blocks: vec![
                BasicBlock::new(in_channels, out_channels, stride, device),
                BasicBlock::new(out_channels, out_channels, 1, device),
            ],
        }
    }

    fn forward(&self, mut input: Tensor<4>) -> Tensor<4> {
        for block in &self.blocks {
            input = block.forward(input);
        }
        input
    }
}

#[derive(Module, Debug)]
pub struct ResNet18 {
    conv1: Conv2d,
    bn1: BatchNorm,
    relu: Relu,
    maxpool: MaxPool2d,
    layer1: LayerBlock,
    layer2: LayerBlock,
    layer3: LayerBlock,
    layer4: LayerBlock,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear,
}

impl ResNet18 {
    pub fn new(num_classes: usize, device: &Device) -> Self {
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };
        Self {
            conv1: Conv2dConfig::new([3, 64], [7, 7])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
                .with_bias(false)
                .with_initializer(initializer)
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            relu: Relu::new(),
            maxpool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(),
            layer1: LayerBlock::new(64, 64, 1, device),
            layer2: LayerBlock::new(64, 128, 2, device),
            layer3: LayerBlock::new(128, 256, 2, device),
            layer4: LayerBlock::new(256, 512, 2, device),
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(512, num_classes).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        let out = self
            .relu
            .forward(self.bn1.forward(self.conv1.forward(input)));
        let out = self.maxpool.forward(out);
        let out = self.layer1.forward(out);
        let out = self.layer2.forward(out);
        let out = self.layer3.forward(out);
        let out = self.layer4.forward(out);
        let out = self.avgpool.forward(out).flatten(1, 3);
        self.fc.forward(out)
    }
}

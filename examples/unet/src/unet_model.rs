use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::ConvTranspose2d;
use burn::nn::conv::ConvTranspose2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::Relu;
use burn::tensor::backend::Backend;
use burn::tensor::Float;
use burn::tensor::Tensor;

/// A convolutional block that performs two consecutive convolutional
/// operations, each followed by a ReLU activation and optional Batch
/// Normalization.
#[derive(Module, Debug)]
pub struct DoubleConv<B: Backend> {
    /// 2d convolutional layer on the input tensor
    conv1: Conv2d<B>,
    /// Batch normalization
    norm1: BatchNorm<B, 2>,
    /// Rectified linear unit
    relu1: Relu,
    /// Intermediate 2d convolutional layer
    conv2: Conv2d<B>,
    /// Intermediate batch normalization
    norm2: BatchNorm<B, 2>,
    /// Final rectified linear unit output
    relu2: Relu,
}

/// Configuration used to construct a new ConvBlock object
#[derive(Config, Debug)]
pub struct DoubleConvConfig {
    /// in_channels: Number of input channels
    in_channels: usize,
    /// out_channels: Number of output channels
    out_channels: usize,
}

impl DoubleConvConfig {
    /// Returns the initialized ConvBlock struct
    pub fn init<B: Backend>(&self, device: &B::Device) -> DoubleConv<B> {
        DoubleConv {
            conv1: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm1: BatchNormConfig::new(self.out_channels).init(device),
            relu1: Relu::new(),
            conv2: Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm2: BatchNormConfig::new(self.out_channels).init(device),
            relu2: Relu::new(),
        }
    }
}

impl<B: Backend> DoubleConv<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        // x shape: [batch_size, in_channels, height, width]
        let x = self.conv1.forward(x);
        let x = self.norm1.forward(x);
        let x = self.relu1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.norm2.forward(x);
        self.relu2.forward(x)
    }
}

// ----------------------------------------------------------------------------

/// Downscaling with maxpool followed by convolutional block.
#[derive(Module, Debug)]
pub struct Down<B: Backend> {
    /// 2d max pooling layer on the input tensor
    max_pool: MaxPool2d,
    conv_block: DoubleConv<B>,
}

/// Configuration used to construct a new ConvBlock object
#[derive(Config, Debug)]
pub struct DownConfig {
    /// in_channels: Number of input channels
    in_channels: usize,
    /// out_channels: Number of output channels
    out_channels: usize,
}

impl DownConfig {
    /// Returns the initialized ConvBlock struct
    pub fn init<B: Backend>(&self, device: &B::Device) -> Down<B> {
        Down {
            max_pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv_block: DoubleConvConfig::new(self.in_channels, self.out_channels).init(device),
        }
    }
}

impl<B: Backend> Down<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        let x = self.max_pool.forward(x);
        self.conv_block.forward(x)
    }
}

// ----------------------------------------------------------------------------

/// Upsampling followed by convolutional block.
#[derive(Module, Debug)]
pub struct Up<B: Backend> {
    /// 2d transposed convolution on the input tensor
    conv_trans: ConvTranspose2d<B>,
    conv_block: DoubleConv<B>,
}

/// Configuration used to construct a new Up object for upsampling
#[derive(Config, Debug)]
pub struct UpConfig {
    /// in_channels: Number of input channels
    in_channels: usize,
    /// out_channels: Number of output channels
    out_channels: usize,
}

impl UpConfig {
    /// Returns the initialized Up struct
    pub fn init<B: Backend>(&self, device: &B::Device) -> Up<B> {
        Up {
            conv_trans: ConvTranspose2dConfig::new([self.in_channels, self.out_channels], [2, 2])
                .with_stride([2, 2])
                .init(device),
            conv_block: DoubleConvConfig::new(self.out_channels, self.out_channels).init(device),
        }
    }
}

impl<B: Backend> Up<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        // TODO: skip connections!
        let x = self.conv_trans.forward(x);
        self.conv_block.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct OutConv<B: Backend> {
    /// 2d convolutional layer on the input tensor
    conv1: Conv2d<B>,
    /// Rectified linear unit
    relu1: Relu,
}

/// Configuration used to construct a new ConvBlock object
#[derive(Config, Debug)]
pub struct OutConvConfig {
    /// in_channels: Number of input channels
    in_channels: usize,
    /// out_channels: Number of output channels
    out_channels: usize,
}

impl OutConvConfig {
    /// Returns the initialized ConvBlock struct
    pub fn init<B: Backend>(&self, device: &B::Device) -> OutConv<B> {
        OutConv {
            conv1: Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1]).init(device),
            relu1: Relu::new(),
        }
    }
}

impl<B: Backend> OutConv<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        // x shape: [batch_size, in_channels, height, width]
        let x = self.conv1.forward(x);
        self.relu1.forward(x)
    }
}

// ----------------------------------------------------------------------------

/// U-net architecture for image segmentation
#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    // Encoder path
    inc: DoubleConv<B>,
    down1: Down<B>,
    down2: Down<B>,
    down3: Down<B>,
    down4: Down<B>,
    // Decoder path
    up4: Up<B>,
    up3: Up<B>,
    up2: Up<B>,
    up1: Up<B>,
    // Final output layer
    outc: OutConv<B>,
}

/// Configuration used to construct a new UNet
#[derive(Config, Debug)]
pub struct UNetConfig;

impl UNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        UNet {
            inc: DoubleConvConfig::new(3, 64).init(device),
            down1: DownConfig::new(64, 128).init(device),
            down2: DownConfig::new(128, 256).init(device),
            down3: DownConfig::new(256, 512).init(device),
            down4: DownConfig::new(512, 1024).init(device),
            up4: UpConfig::new(1024, 512).init(device),
            up3: UpConfig::new(512, 256).init(device),
            up2: UpConfig::new(256, 128).init(device),
            up1: UpConfig::new(128, 64).init(device),
            outc: OutConvConfig::new(64, 1).init(device),
        }
    }
}

impl<B: Backend> UNet<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        let x0 = self.inc.forward(x); // [b, 64, h, w]
        let x1 = self.down1.forward(x0.clone()); // [b, 128, h/2, w/2]
        let x2 = self.down2.forward(x1.clone()); // [b, 256, h/4, w/4]
        let x3 = self.down3.forward(x2.clone()); // [b, 512, h/8, w/8]
        let x4 = self.down4.forward(x3.clone()); // [b, 1024, h/16, w/16]
        let x3 = self.up4.forward(x4.clone()) + x3; // [b, 512, h/8, w/8]
        let x2 = self.up3.forward(x3) + x2; // [b, 256, h/4, w/4]
        let x1 = self.up2.forward(x2) + x1; // [b, 128, h/2, w/2]
        let x0 = self.up1.forward(x1) + x0; // [b, 64, h, w]
        self.outc.forward(x0) // [b, 1, h, w]
    }
}

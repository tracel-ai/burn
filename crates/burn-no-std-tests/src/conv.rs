// Originally copied from the burn/examples/mnist package

use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{Device, Tensor},
};

#[derive(Module, Debug)]
pub struct ConvBlock {
    conv: nn::conv::Conv2d,
    pool: nn::pool::MaxPool2d,
    activation: nn::Gelu,
}

#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    channels: [usize; 2],
    #[config(default = "[3, 3]")]
    kernel_size: [usize; 2],
}

impl ConvBlock {
    pub fn new(config: &ConvBlockConfig, device: &Device) -> Self {
        let conv = nn::conv::Conv2dConfig::new(config.channels, config.kernel_size)
            .with_padding(nn::PaddingConfig2d::Same)
            .init(device);
        let pool = nn::pool::MaxPool2dConfig::new(config.kernel_size)
            .with_strides([1, 1])
            .with_padding(nn::PaddingConfig2d::Same)
            .init();
        let activation = nn::Gelu::new();

        Self {
            conv,
            pool,
            activation,
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(input.clone());
        let x = self.pool.forward(x);
        let x = self.activation.forward(x);

        (x + input) / 2.0
    }
}

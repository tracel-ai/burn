// Originally copied from the burn/examples/mnist package

use crate::{
    conv::{ConvBlock, ConvBlockConfig},
    mlp::{Mlp, MlpConfig},
};

use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{backend::Backend, Tensor},
};

#[derive(Config)]
pub struct MnistConfig {
    #[config(default = 42)]
    pub seed: u64,

    pub mlp: MlpConfig,

    #[config(default = 784)]
    pub input_size: usize,

    #[config(default = 10)]
    pub output_size: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    mlp: Mlp<B>,
    conv: ConvBlock<B>,
    input: nn::Linear<B>,
    output: nn::Linear<B>,
    num_classes: usize,
}

impl<B: Backend> Model<B> {
    pub fn new(config: &MnistConfig) -> Self {
        let mlp = Mlp::new(&config.mlp);
        let input = nn::LinearConfig::new(config.input_size, config.mlp.d_model).init();
        let output = nn::LinearConfig::new(config.mlp.d_model, config.output_size).init();
        let conv = ConvBlock::new(&ConvBlockConfig::new([1, 1]));

        Self {
            mlp,
            conv,
            output,
            input,
            num_classes: config.output_size,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();

        let x = input.reshape([batch_size, 1, height, width]).detach();
        let x = self.conv.forward(x);
        let x = x.reshape([batch_size, height * width]);

        let x = self.input.forward(x);
        let x = self.mlp.forward(x);

        self.output.forward(x)
    }
}

use burn::{
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu},
    prelude::*,
};

/// The model as it trains on one device: an input projection, a stack of identical blocks, a final
/// norm and an output projection.
#[derive(Module, Debug)]
pub struct Model {
    input: Linear,
    blocks: Vec<Block>,
    norm: LayerNorm,
    output: Linear,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 16)]
    pub features: usize,
    #[config(default = 256)]
    pub hidden: usize,
    #[config(default = 8)]
    pub blocks: usize,
    #[config(default = 1)]
    pub outputs: usize,
}

impl ModelConfig {
    pub fn init(&self, device: &Device) -> Model {
        Model {
            input: LinearConfig::new(self.features, self.hidden).init(device),
            blocks: (0..self.blocks)
                .map(|_| Block::new(self.hidden, device))
                .collect(),
            norm: LayerNormConfig::new(self.hidden).init(device),
            output: LinearConfig::new(self.hidden, self.outputs).init(device),
        }
    }
}

impl Model {
    pub fn forward(&self, features: Tensor<2>) -> Tensor<2> {
        let hidden = self
            .blocks
            .iter()
            .fold(self.input.forward(features), |hidden, block| {
                block.forward(hidden)
            });
        self.output.forward(self.norm.forward(hidden))
    }
}

/// A residual block, the same in both shapes of the model.
#[derive(Module, Debug)]
pub struct Block {
    norm: LayerNorm,
    linear: Linear,
    activation: Relu,
}

impl Block {
    /// The `burn-nn` layers initialize lazily, so nothing is allocated on `device` yet.
    pub fn new(hidden: usize, device: &Device) -> Self {
        Self {
            norm: LayerNormConfig::new(hidden).init(device),
            linear: LinearConfig::new(hidden, hidden).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, hidden: Tensor<2>) -> Tensor<2> {
        let normed = self.norm.forward(hidden.clone());
        self.activation.forward(self.linear.forward(normed)) + hidden
    }
}

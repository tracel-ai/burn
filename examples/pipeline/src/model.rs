use burn::{
    module::pipeline::{Pipeline, PipelineLayout, PipelinePlacement},
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu},
    prelude::*,
};

/// One input projection, a stack of identical blocks, one output head: the shape pipeline
/// parallelism exists for, where the blocks are what gets shared out across devices.
#[derive(Module, Debug)]
pub struct Model {
    input: Linear,
    blocks: Vec<Block>,
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
    /// The `burn-nn` layers initialize lazily, so nothing is allocated on `device`: a model built
    /// here and then placed never exists whole on one device.
    pub fn init(&self, device: &Device) -> Model {
        Model {
            input: LinearConfig::new(self.features, self.hidden).init(device),
            blocks: (0..self.blocks)
                .map(|_| Block {
                    norm: LayerNormConfig::new(self.hidden).init(device),
                    linear: LinearConfig::new(self.hidden, self.hidden).init(device),
                    activation: Relu::new(),
                })
                .collect(),
            output: LinearConfig::new(self.hidden, self.outputs).init(device),
        }
    }
}

impl Model {
    /// Each segment's parameter count and device.
    pub fn print_placement(&self, placement: &PipelinePlacement) {
        println!("placement:");
        println!(
            "  input    {:>8} params  {:?}",
            self.input.num_params(),
            placement.input
        );
        for (index, device) in placement.blocks.iter().enumerate() {
            let params = self.blocks[index].num_params();
            println!("  block {index:<2} {params:>8} params  {device:?}");
        }
        println!(
            "  output   {:>8} params  {:?}",
            self.output.num_params(),
            placement.output
        );
    }
}

#[derive(Module, Debug)]
pub struct Block {
    norm: LayerNorm,
    linear: Linear,
    activation: Relu,
}

impl Block {
    fn forward(&self, hidden: Tensor<2>) -> Tensor<2> {
        let normed = self.norm.forward(hidden.clone());
        self.activation.forward(self.linear.forward(normed)) + hidden
    }
}

impl Pipeline for Model {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type Carry = Tensor<2>;

    fn layout(&self) -> PipelineLayout {
        PipelineLayout::new()
            .input(&self.input)
            .blocks(&self.blocks)
            .output(&self.output)
    }

    fn forward_input(&self, features: Tensor<2>) -> Tensor<2> {
        self.input.forward(features)
    }

    fn forward_block(&self, index: usize, hidden: Tensor<2>) -> Tensor<2> {
        self.blocks[index].forward(hidden)
    }

    fn forward_output(&self, hidden: Tensor<2>) -> Tensor<2> {
        self.output.forward(hidden)
    }
}

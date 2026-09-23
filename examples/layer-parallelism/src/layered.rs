use burn::{
    module::parallel::{DistributedLayer, LayerParallelism, LayerPlacement},
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
    store::BurnpackStore,
};

use crate::model::{Block, ModelConfig};

/// [`Model`](crate::model::Model) reshaped to be split across devices: the input projection becomes
/// an embedding layer, and the final norm joins the output projection in a head, so the model is an
/// input layer, hidden layers and an output layer.
#[derive(Module, Debug)]
pub struct LayeredModel {
    embedding: Embedding,
    hiddens: Vec<Block>,
    head: Head,
}

impl LayeredModel {
    /// Each layer is built on the device `placement` gives it, so the model never exists whole on
    /// one device.
    pub fn new(config: &ModelConfig, placement: &LayerPlacement) -> Self {
        Self {
            embedding: Embedding {
                linear: LinearConfig::new(config.features, config.hidden).init(&placement.input),
            },
            hiddens: placement
                .hidden
                .iter()
                .map(|device| Block::new(config.hidden, device))
                .collect(),
            head: Head {
                norm: LayerNormConfig::new(config.hidden).init(&placement.output),
                linear: LinearConfig::new(config.hidden, config.outputs).init(&placement.output),
            },
        }
    }

    /// Read a checkpoint saved from [`Model`](crate::model::Model) under this model's keys.
    pub fn remap_from_single_device(store: BurnpackStore) -> BurnpackStore {
        store
            .with_remap_pattern(r"^input\.", "embedding.linear.")
            .with_remap_pattern(r"^blocks\.", "hiddens.")
            .with_remap_pattern(r"^norm\.", "head.norm.")
            .with_remap_pattern(r"^output\.", "head.linear.")
    }

    /// Each layer's parameter count and device.
    pub fn print_placement(&self, placement: &LayerPlacement) {
        println!("placement:");
        println!(
            "  embedding {:>8} params  {:?}",
            self.embedding.num_params(),
            placement.input
        );
        for (index, device) in placement.hidden.iter().enumerate() {
            let params = self.hiddens[index].num_params();
            println!("  block {index:<3} {params:>8} params  {device:?}");
        }
        println!(
            "  head      {:>8} params  {:?}",
            self.head.num_params(),
            placement.output
        );
    }
}

impl LayerParallelism for LayeredModel {
    type InputLayer = Embedding;
    type HiddenLayer = Block;
    type OutputLayer = Head;

    fn layer_input(&self) -> &Embedding {
        &self.embedding
    }

    fn layer_hidden(&self, index: usize) -> Option<&Block> {
        self.hiddens.get(index)
    }

    fn layer_output(&self) -> &Head {
        &self.head
    }
}

#[derive(Module, Debug)]
pub struct Embedding {
    linear: Linear,
}

impl DistributedLayer for Embedding {
    type Input = Tensor<2>;
    type Output = Tensor<2>;

    fn forward(&self, features: Tensor<2>) -> Tensor<2> {
        self.linear.forward(features)
    }
}

impl DistributedLayer for Block {
    type Input = Tensor<2>;
    type Output = Tensor<2>;

    fn forward(&self, hidden: Tensor<2>) -> Tensor<2> {
        Block::forward(self, hidden)
    }
}

#[derive(Module, Debug)]
pub struct Head {
    norm: LayerNorm,
    linear: Linear,
}

impl DistributedLayer for Head {
    type Input = Tensor<2>;
    type Output = Tensor<2>;

    fn forward(&self, hidden: Tensor<2>) -> Tensor<2> {
        self.linear.forward(self.norm.forward(hidden))
    }
}

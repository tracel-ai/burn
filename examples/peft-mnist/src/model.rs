use burn::{
    nn::{Linear, LinearConfig, Relu, loss::CrossEntropyLossConfig},
    prelude::*,
    tensor::Int,
    train::ClassificationOutput,
};
use burn_peft::{DoRAConfig, DoRALinear, LoRAConfig, LoRALinear};

/// Simple MLP for MNIST classification
#[derive(Module, Debug)]
pub struct SimpleMLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> SimpleMLP<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(784, 512).init(device),
            fc2: LinearConfig::new(512, 256).init(device),
            fc3: LinearConfig::new(256, 10).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.flatten(1, 1);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        self.fc3.forward(x)
    }
}

/// MLP with LoRA adapters on all linear layers
#[derive(Module, Debug)]
pub struct LoRAMLP<B: Backend> {
    fc1: LoRALinear<B>,
    fc2: LoRALinear<B>,
    fc3: LoRALinear<B>,
    activation: Relu,
}

impl<B: Backend> LoRAMLP<B> {
    /// Convert a pretrained MLP to LoRA
    pub fn from_pretrained(
        pretrained: SimpleMLP<B>,
        rank: usize,
        alpha: f64,
        device: &B::Device,
    ) -> Self {
        let config1 = LoRAConfig::new(784, 512).with_rank(rank).with_alpha(alpha);
        let config2 = LoRAConfig::new(512, 256).with_rank(rank).with_alpha(alpha);
        let config3 = LoRAConfig::new(256, 10).with_rank(rank).with_alpha(alpha);

        Self {
            fc1: config1.init_with_base_weight(
                pretrained.fc1.weight.val(),
                pretrained.fc1.bias.map(|b| b.val()),
                device,
            ),
            fc2: config2.init_with_base_weight(
                pretrained.fc2.weight.val(),
                pretrained.fc2.bias.map(|b| b.val()),
                device,
            ),
            fc3: config3.init_with_base_weight(
                pretrained.fc3.weight.val(),
                pretrained.fc3.bias.map(|b| b.val()),
                device,
            ),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.flatten(1, 1);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        self.fc3.forward(x)
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let batch_size = images.dims()[0];
        let output = self.forward(images.reshape([batch_size, 784]));
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }

    /// Merge all LoRA adapters into base weights for inference
    pub fn merge_weights(&mut self) {
        self.fc1.merge_weights();
        self.fc2.merge_weights();
        self.fc3.merge_weights();
    }

    /// Unmerge all LoRA adapters from base weights
    pub fn unmerge_weights(&mut self) {
        self.fc1.unmerge_weights();
        self.fc2.unmerge_weights();
        self.fc3.unmerge_weights();
    }
}

/// MLP with DoRA adapters
#[derive(Module, Debug)]
pub struct DoRAMLP<B: Backend> {
    fc1: DoRALinear<B>,
    fc2: DoRALinear<B>,
    fc3: DoRALinear<B>,
    activation: Relu,
}

impl<B: Backend> DoRAMLP<B> {
    pub fn from_pretrained(pretrained: SimpleMLP<B>, rank: usize, device: &B::Device) -> Self {
        let config1 = DoRAConfig::new(784, 512).with_rank(rank);
        let config2 = DoRAConfig::new(512, 256).with_rank(rank);
        let config3 = DoRAConfig::new(256, 10).with_rank(rank);

        Self {
            fc1: config1.init_with_base_weight(
                pretrained.fc1.weight.val(),
                pretrained.fc1.bias.map(|b| b.val()),
                device,
            ),
            fc2: config2.init_with_base_weight(
                pretrained.fc2.weight.val(),
                pretrained.fc2.bias.map(|b| b.val()),
                device,
            ),
            fc3: config3.init_with_base_weight(
                pretrained.fc3.weight.val(),
                pretrained.fc3.bias.map(|b| b.val()),
                device,
            ),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.flatten(1, 1);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        self.fc3.forward(x)
    }
}

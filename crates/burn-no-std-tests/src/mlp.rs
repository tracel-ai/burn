// Originally copied from the burn/examples/mnist package

use alloc::vec::Vec;

use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{backend::Backend, Tensor},
};

/// Configuration to create a [Multilayer Perceptron](Mlp) layer.
#[derive(Config)]
pub struct MlpConfig {
    /// The number of layers.
    #[config(default = 3)]
    pub num_layers: usize,
    /// The dropout rate.
    #[config(default = 0.5)]
    pub dropout: f64,
    /// The size of each layer.
    #[config(default = 256)]
    pub d_model: usize,
}

/// Multilayer Perceptron module.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linears: Vec<nn::Linear<B>>,
    dropout: nn::Dropout,
    activation: nn::Relu,
}

impl<B: Backend> Mlp<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &MlpConfig, device: &B::Device) -> Self {
        let mut linears = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            linears.push(nn::LinearConfig::new(config.d_model, config.d_model).init(device));
        }

        Self {
            linears,
            dropout: nn::DropoutConfig::new(0.3).init(),
            activation: nn::Relu::new(),
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, d_model]`
    /// - output: `[batch_size, d_model]`
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        for linear in self.linears.iter() {
            x = linear.forward(x);
            x = self.dropout.forward(x);
            x = self.activation.forward(x);
        }

        x
    }
}

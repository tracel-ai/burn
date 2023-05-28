use crate as burn;

use burn_tensor::Tensor;
use burn_tensor::backend::Backend;
use crate::module::Module;
use crate::nn::Linear;
use crate::nn::LinearConfig;
use crate::nn::Initializer;

#[derive(Module, Debug)]
pub struct GateController<B: Backend> {
    /// Represents the affine transformation applied to input vector
    input_transform: Linear<B>,
    /// Represents the affine transformation applied to the hidden state
    hidden_transform: Linear<B>,
}

impl<B: Backend> GateController<B> {
    pub fn new(d_input: usize, d_output: usize, bias: bool, initializer: Initializer) -> GateController<B> {
        GateController { 
            input_transform: LinearConfig{ d_input: d_input, d_output: d_output, bias: bias, initializer: initializer.clone() }.init(), 
            hidden_transform: LinearConfig{ d_input: d_input, d_output: d_output, bias: bias, initializer: initializer.clone() }.init(), 
        }
    }

    pub fn get_input_weight(&self) -> Tensor<B, 2> {
        self.input_transform.get_weight()
    }

    pub fn get_hidden_weight(&self) -> Tensor<B, 2>{
        self.hidden_transform.get_weight()
    }

    pub fn get_input_bias(&self) -> Option<Tensor<B, 1>> {
        self.input_transform.get_bias()
    }

    pub fn get_hidden_bias(&self) -> Option<Tensor<B, 1>> {
        self.hidden_transform.get_bias()
    }
}


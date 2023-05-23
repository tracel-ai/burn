use crate as burn;

use crate::module::Param;
use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

use super::{Initializer, Linear, LinearConfig};

#[derive(Config)]
pub struct LSTMConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the LSTM transformation
    pub bias: bool,
    /// The type of function used to initialize LSTM parameters
    #[config(default = "Initializer::UniformDefault")]
    pub initializer: Initializer,
    /// The batch size
    pub batch_size: usize,
}

/// The LSTM module. This implementation is for a unidirectional, stateful, LSTM.
#[derive(Module, Debug)]
pub struct LSTM<B: Backend> {
    input_gate: Linear<B>,
    forget_gate: Linear<B>,
    output_gate: Linear<B>,
    cell_gate: Linear<B>,
    hidden_state: Option<Param<Tensor<B, 2>>>,
    cell_state: Option<Param<Tensor<B, 2>>>,
}

impl LSTMConfig {
    /// Initialize a new [lstm](LSTM) module
    pub fn init<B: Backend>(&self) -> LSTM<B> {
        let d_output = self.d_hidden;

        let input_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();
        let forget_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();
        let output_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();
        let cell_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();

        let hidden_state = Some(Param::from(Tensor::zeros([self.batch_size, self.d_hidden])));
        let cell_state = Some(Param::from(Tensor::zeros([self.batch_size, self.d_hidden])));

        LSTM {
            input_gate,
            forget_gate,
            output_gate,
            cell_gate,
            hidden_state,
            cell_state,
        }
    }
}

impl<B: Backend> LSTM<B> {
    /// Applies the forward pass on the input tensor
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>, state: Option<(Tensor<B, D>, Tensor<B, D>)>) -> (Tensor<B, D>, Tensor<B, D>, Tensor<B, D>) {

    }
}
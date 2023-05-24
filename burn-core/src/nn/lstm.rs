use burn_tensor::activation;

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
    /// If a bias should be applied during the LSTM transformation.
    pub bias: bool,
    /// LSTM bias for gate controllers usually initialized to ones
    /// to prevent forgetting at the beginning of training.
    #[config(default = "Initializer::Ones")]
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
    batch_size: usize,
    d_hidden: usize,
}

impl LSTMConfig {
    /// Initialize a new [lstm](LSTM) module
    pub fn init<B: Backend>(&self) -> LSTM<B> {
        let d_output = self.d_hidden;

        let input_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();
        let forget_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();
        let output_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();
        let cell_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();

        /// Do these belong here?
        let hidden_state = Some(Param::from(Tensor::zeros([self.batch_size, self.d_hidden])));
        let cell_state = Some(Param::from(Tensor::zeros([self.batch_size, self.d_hidden])));

        LSTM {
            input_gate,
            forget_gate,
            output_gate,
            cell_gate,
            hidden_state,
            cell_state,
            batch_size: self.batch_size,
            d_hidden: self.d_hidden,
        }
    }
}

impl<B: Backend> LSTM<B> {
    /// Applies the forward pass on the input tensor. In this implementation,
    /// the LSTM returns only the last time step's output, hence <B, 2> for
    /// all output tensors, with dimensions [batch_size, hidden_size]
    /// 
    /// inputs: 
    ///     input - the input tensor
    ///     state -  the cell state and hidden state
    /// outputs:
    ///     3 tensors, one for the cell state, one for the hidden state,
    ///     and one for the result = hidden state. 
    pub fn forward<const D: usize>(&self, input: Tensor<B, 2>, state: Option<(Tensor<B, 2>, Tensor<B, 2>)>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (cell_state, hidden_state) = match state {
            // If state is provided
            Some((cell_state, hidden_state)) => (cell_state, hidden_state),
            None => (
                self.cell_state.as_ref().unwrap().val().clone(),
                self.hidden_state.as_ref().unwrap().val().clone(),
            ),
        };
        
        let input_product: Tensor<B, 2> = input.matmul(self.forget_gate.get_weight().unsqueeze());
        let hidden_product: Tensor<B, 2> = hidden_state.matmul(self.forget_gate.get_weight().unsqueeze());
        
        match &self.hidden_state {
            Some(hidden_state) => hidden_state.val().matmul(self.forget_gate.get_weight().unsqueeze()),
            None => Tensor::zeros([self.batch_size, self.d_hidden]),
        };

        let biased_input_sum = match &self.forget_gate.get_bias() {
            Some(bias) => input_product + hidden_product + bias.clone().unsqueeze(),
            None => input_product + hidden_product,
        };
        let forget_values = activation::sigmoid(biased_input_sum);

        (forget_values.clone(), forget_values.clone(), forget_values.clone()) // to match expected return, will update later
    }

    pub fn reset_states(&mut self) {
        self.hidden_state = Some(Param::from(Tensor::zeros([self.batch_size, self.d_hidden])));
        self.cell_state = Some(Param::from(Tensor::zeros([self.batch_size, self.d_hidden])));
    }
}
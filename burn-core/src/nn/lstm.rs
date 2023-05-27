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
    /// LSTM initializer, should probably be Xavier or small random numbers
    #[config(default = "Initializer::Uniform(0.0, 1.0)")]
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

    pub fn init_with_states<B: Backend>(&self, hidden: Tensor<B,2>, cell: Tensor<B, 2>) -> LSTM<B> {
        let d_output = self.d_hidden;

        let input_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();
        let forget_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();
        let output_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();
        let cell_gate = LinearConfig{ d_input: self.d_input, d_output, bias: self.bias, initializer: self.initializer.clone() }.init();

        let hidden_state = Some(Param::from(hidden));
        let cell_state = Some(Param::from(cell));

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
    pub fn forward<const D: usize>(&mut self, input: Tensor<B, 2>, state: Option<(Tensor<B, 2>, Tensor<B, 2>)>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (mut cell_state, mut hidden_state) = match state {
            // If state is provided
            Some((cell_state, hidden_state)) => (cell_state, hidden_state),
            None => (
                self.cell_state.as_ref().unwrap().val().clone(),
                self.hidden_state.as_ref().unwrap().val().clone(),
            ),
        };
    
        // f(orget)g(ate) tensors 
        let biased_fg_input_sum = self.gate_product(&input, &hidden_state, &self.forget_gate);
        let forget_values = activation::sigmoid(biased_fg_input_sum); // to multiply with cell state

        // i(nput)g(ate) tensors
        let biased_ig_input_sum = self.gate_product(&input, &hidden_state, &self.input_gate);
        let add_values = activation::sigmoid(biased_ig_input_sum);

        // o(utput)g(ate) tensors
        let biased_og_input_sum = self.gate_product(&input, &hidden_state, &self.output_gate);
        let output_values = activation::sigmoid(biased_og_input_sum);

         // c(ell)g(ate) tensors
        let biased_cg_input_sum = self.gate_product(&input, &hidden_state, &self.cell_gate);
        let candidate_cell_values = biased_cg_input_sum.tanh();
        let candidate_cell_values_clone = candidate_cell_values.clone();

        cell_state = forget_values * cell_state + add_values * candidate_cell_values;
        hidden_state = output_values * candidate_cell_values_clone;

        self.cell_state = Some(Param::from(cell_state.clone()));
        self.hidden_state = Some(Param::from(hidden_state.clone()));

        (cell_state.clone(), hidden_state.clone(), hidden_state.clone()) // should hidden_state be returned twice?
    }

    /// Helper function for performing weighted matrix product for a gate and adds
    /// bias, if any.
    /// 
    ///  Mathematically, performs `Wx*X + Wh*H + b`, where:
    ///     Wx = weight matrix for the connection to input vector X
    ///     Wh = weight matrix for the connection to hidden state H
    ///     X = input vector
    ///     H = hidden state
    ///     b = bias terms
    fn gate_product(&self, input: &Tensor<B, 2>, hidden: &Tensor<B, 2>, gate: &Linear<B>) -> Tensor<B, 2> {
        let input_product = input.clone().matmul(gate.get_weight().unsqueeze());
        let hidden_product = hidden.clone().matmul(gate.get_weight().unsqueeze());

        match &gate.get_bias() {
            Some(bias) => input_product + hidden_product + bias.clone().unsqueeze(),
            None => input_product + hidden_product,
        }
    }

    /// Reset the hidden and cell states of the LSTM cell. This should be done
    /// after each full pass through a sequence.
    pub fn reset_states(&mut self) {
        self.hidden_state = Some(Param::from(Tensor::zeros([self.batch_size, self.d_hidden])));
        self.cell_state = Some(Param::from(Tensor::zeros([self.batch_size, self.d_hidden])));
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::Data;

    use super::*;
    use crate::TestBackend;
    // use burn_tensor::Data;
    // use libm::sqrt;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = LSTMConfig::new(5, 5, false, 2);
        let lstm = config.init::<TestBackend>();

        lstm.input_gate.get_weight().to_data().assert_in_range(0.0, 1.0);
        lstm.forget_gate.get_weight().to_data().assert_in_range(0.0, 1.0);
        lstm.output_gate.get_weight().to_data().assert_in_range(0.0, 1.0);
        lstm.cell_gate.get_weight().to_data().assert_in_range(0.0, 1.0);
    }

    #[test]
    fn pass_states() {
        TestBackend::seed(0);

        let test_hidden_state = Tensor::<TestBackend, 2>::from_floats([[0.1, 0.2, 0.3, 0.4]]);
        let test_cell_state = Tensor::<TestBackend, 2>::from_floats([[0.4, 0.5, 0.6, 0.7]]);

        let config = LSTMConfig::new(4, 4, false, 2);
        let lstm = config.init_with_states::<TestBackend>(test_hidden_state, test_cell_state);

        assert_eq!(lstm.hidden_state.as_ref().unwrap().val().to_data(), Data::from([[0.1, 0.2, 0.3, 0.4]]));
        assert_eq!(lstm.cell_state.as_ref().unwrap().val().to_data(), Data::from([[0.4, 0.5, 0.6, 0.7]]));
    }

    #[test]
    fn reset_hidden_and_cell_state() {
        TestBackend::seed(0);

        let test_hidden_state = Tensor::<TestBackend, 2>::from_floats([[0.1, 0.2, 0.3, 0.4]]);
        let test_cell_state = Tensor::<TestBackend, 2>::from_floats([[0.4, 0.5, 0.6, 0.7]]);

        let config = LSTMConfig::new(4, 4, false, 1);
        let mut lstm = config.init_with_states::<TestBackend>(test_hidden_state, test_cell_state);

        assert_eq!(lstm.hidden_state.as_ref().unwrap().val().to_data(), Data::from([[0.1, 0.2, 0.3, 0.4]]));
        assert_eq!(lstm.cell_state.as_ref().unwrap().val().to_data(), Data::from([[0.4, 0.5, 0.6, 0.7]]));

        lstm.reset_states();

        assert_eq!(lstm.hidden_state.as_ref().unwrap().val().to_data(), Data::from([[0.0, 0.0, 0.0, 0.0]]));
        assert_eq!(lstm.cell_state.as_ref().unwrap().val().to_data(), Data::from([[0.0, 0.0, 0.0, 0.0]]));
    }

    #[test]
    fn test_forward() {
        // Run same input through PyTorch or TensorFlow and compare?
        todo!()
    }
}
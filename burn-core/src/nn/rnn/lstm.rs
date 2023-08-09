use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::nn::rnn::gate_controller;
use crate::nn::Initializer;
use crate::nn::LinearConfig;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::activation;

use super::gate_controller::GateController;

/// The configuration for a [lstm](Lstm) module.
#[derive(Config)]
pub struct LstmConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the Lstm transformation.
    pub bias: bool,
    /// Lstm initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
    /// The batch size.
    pub batch_size: usize,
}

/// The Lstm module. This implementation is for a unidirectional, stateless, Lstm.
#[derive(Module, Debug)]
pub struct Lstm<B: Backend> {
    input_gate: GateController<B>,
    forget_gate: GateController<B>,
    output_gate: GateController<B>,
    cell_gate: GateController<B>,
    batch_size: usize,
    d_hidden: usize,
}

impl LstmConfig {
    /// Initialize a new [lstm](Lstm) module.
    pub fn init<B: Backend>(&self) -> Lstm<B> {
        let d_output = self.d_hidden;

        let input_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
        );
        let forget_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
        );
        let output_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
        );
        let cell_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
        );

        Lstm {
            input_gate,
            forget_gate,
            output_gate,
            cell_gate,
            batch_size: self.batch_size,
            d_hidden: self.d_hidden,
        }
    }

    /// Initialize a new [lstm](Lstm) module with a [record](LstmRecord).
    pub fn init_with<B: Backend>(&self, record: LstmRecord<B>) -> Lstm<B> {
        let linear_config = LinearConfig {
            d_input: self.d_input,
            d_output: self.d_hidden,
            bias: self.bias,
            initializer: self.initializer.clone(),
        };

        Lstm {
            input_gate: gate_controller::GateController::new_with(
                &linear_config,
                record.input_gate,
            ),
            forget_gate: gate_controller::GateController::new_with(
                &linear_config,
                record.forget_gate,
            ),
            output_gate: gate_controller::GateController::new_with(
                &linear_config,
                record.output_gate,
            ),
            cell_gate: gate_controller::GateController::new_with(&linear_config, record.cell_gate),
            batch_size: self.batch_size,
            d_hidden: self.d_hidden,
        }
    }
}

impl<B: Backend> Lstm<B> {
    /// Applies the forward pass on the input tensor. This LSTM implementation
    /// returns the cell state and hidden state for each element in a sequence (i.e., across `seq_length`),
    /// producing 3-dimensional tensors where the dimensions represent [batch_size, sequence_length, hidden_size].
    ///
    /// Parameters:
    ///     batched_input: The input tensor of shape [batch_size, sequence_length, input_size].
    ///     state: An optional tuple of tensors representing the initial cell state and hidden state.
    ///            Each state tensor has shape [batch_size, hidden_size].
    ///            If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// Returns:
    ///     A tuple of tensors, where the first tensor represents the cell states and
    ///     the second tensor represents the hidden states for each sequence element.
    ///     Both output tensors have the shape [batch_size, sequence_length, hidden_size].
    pub fn forward(
        &mut self,
        batched_input: Tensor<B, 3>,
        state: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let seq_length = batched_input.shape().dims[1];
        let mut batched_cell_state = Tensor::zeros([self.batch_size, seq_length, self.d_hidden]);
        let mut batched_hidden_state = Tensor::zeros([self.batch_size, seq_length, self.d_hidden]);

        let (mut cell_state, mut hidden_state) = match state {
            Some((cell_state, hidden_state)) => (cell_state, hidden_state),
            None => (
                Tensor::zeros([self.batch_size, self.d_hidden]),
                Tensor::zeros([self.batch_size, self.d_hidden]),
            ),
        };

        for t in 0..seq_length {
            let indices = Tensor::arange(t..t + 1);
            let input_t = batched_input.clone().select(1, indices).squeeze(1);
            // f(orget)g(ate) tensors
            let biased_fg_input_sum = self.gate_product(&input_t, &hidden_state, &self.forget_gate);
            let forget_values = activation::sigmoid(biased_fg_input_sum); // to multiply with cell state

            // i(nput)g(ate) tensors
            let biased_ig_input_sum = self.gate_product(&input_t, &hidden_state, &self.input_gate);
            let add_values = activation::sigmoid(biased_ig_input_sum);

            // o(output)g(ate) tensors
            let biased_og_input_sum = self.gate_product(&input_t, &hidden_state, &self.output_gate);
            let output_values = activation::sigmoid(biased_og_input_sum);

            // c(ell)g(ate) tensors
            let biased_cg_input_sum = self.gate_product(&input_t, &hidden_state, &self.cell_gate);
            let candidate_cell_values = biased_cg_input_sum.tanh();

            cell_state = forget_values * cell_state.clone() + add_values * candidate_cell_values;
            hidden_state = output_values * cell_state.clone().tanh();

            // store the state for this timestep
            batched_cell_state = batched_cell_state.slice_assign(
                [0..self.batch_size, t..(t + 1), 0..self.d_hidden],
                cell_state.clone().unsqueeze(),
            );
            batched_hidden_state = batched_hidden_state.slice_assign(
                [0..self.batch_size, t..(t + 1), 0..self.d_hidden],
                hidden_state.clone().unsqueeze(),
            );
        }

        (batched_cell_state, batched_hidden_state)
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
    fn gate_product(
        &self,
        input: &Tensor<B, 2>,
        hidden: &Tensor<B, 2>,
        gate: &GateController<B>,
    ) -> Tensor<B, 2> {
        let input_product = input.clone().matmul(gate.input_transform.weight.val());
        let hidden_product = hidden.clone().matmul(gate.hidden_transform.weight.val());

        let input_bias = gate
            .input_transform
            .bias
            .as_ref()
            .map(|bias_param| bias_param.val());
        let hidden_bias = gate
            .hidden_transform
            .bias
            .as_ref()
            .map(|bias_param| bias_param.val());

        match (input_bias, hidden_bias) {
            (Some(input_bias), Some(hidden_bias)) => {
                input_product + input_bias.unsqueeze() + hidden_product + hidden_bias.unsqueeze()
            }
            (Some(input_bias), None) => input_product + input_bias.unsqueeze() + hidden_product,
            (None, Some(hidden_bias)) => input_product + hidden_product + hidden_bias.unsqueeze(),
            (None, None) => input_product + hidden_product,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{module::Param, nn::LinearRecord, TestBackend};
    use burn_tensor::Data;

    #[test]
    fn test_with_uniform_initializer() {
        TestBackend::seed(0);

        let config = LstmConfig::new(5, 5, false, 2)
            .with_initializer(Initializer::Uniform { min: 0.0, max: 1.0 });
        let lstm = config.init::<TestBackend>();

        let gate_to_data =
            |gate: GateController<TestBackend>| gate.input_transform.weight.val().to_data();

        gate_to_data(lstm.input_gate).assert_within_range(0..1);
        gate_to_data(lstm.forget_gate).assert_within_range(0..1);
        gate_to_data(lstm.output_gate).assert_within_range(0..1);
        gate_to_data(lstm.cell_gate).assert_within_range(0..1);
    }

    /// Test forward pass with simple input vector.
    ///
    /// f_t = sigmoid(0.7*0.1 + 0.7*0) = sigmoid(0.07) = 0.5173928
    /// i_t = sigmoid(0.5*0.1 + 0.5*0) = sigmoid(0.05) = 0.5123725
    /// o_t = sigmoid(1.1*0.1 + 1.1*0) = sigmoid(0.11) = 0.5274723
    /// c_t = tanh(0.9*0.1 + 0.9*0) = tanh(0.09) = 0.0892937

    /// C_t = f_t * 0 + i_t * c_t = 0 + 0.5123725 * 0.0892937 = 0.04575243
    /// h_t = o_t * tanh(C_t) = 0.5274723 * tanh(0.04575243) = 0.5274723 * 0.04568173 = 0.024083648
    #[test]
    fn test_forward_single_input_single_feature() {
        TestBackend::seed(0);
        let config = LstmConfig::new(1, 1, false, 1);
        let mut lstm = config.init::<TestBackend>();

        fn create_gate_controller(
            weights: f32,
            biases: f32,
            d_input: usize,
            d_output: usize,
            bias: bool,
            initializer: Initializer,
        ) -> GateController<TestBackend> {
            let record = LinearRecord {
                weight: Param::from(Tensor::from_data(Data::from([[weights]]))),
                bias: Some(Param::from(Tensor::from_data(Data::from([biases])))),
            };
            gate_controller::GateController::create_with_weights(
                d_input,
                d_output,
                bias,
                initializer,
                record.clone(),
                record,
            )
        }

        lstm.input_gate = create_gate_controller(
            0.5,
            0.0,
            1,
            1,
            false,
            Initializer::XavierUniform { gain: 1.0 },
        );
        lstm.forget_gate = create_gate_controller(
            0.7,
            0.0,
            1,
            1,
            false,
            Initializer::XavierUniform { gain: 1.0 },
        );
        lstm.cell_gate = create_gate_controller(
            0.9,
            0.0,
            1,
            1,
            false,
            Initializer::XavierUniform { gain: 1.0 },
        );
        lstm.output_gate = create_gate_controller(
            1.1,
            0.0,
            1,
            1,
            false,
            Initializer::XavierUniform { gain: 1.0 },
        );

        // single timestep with single feature
        let input = Tensor::<TestBackend, 3>::from_data(Data::from([[[0.1]]]));

        let (cell_state_batch, hidden_state_batch) = lstm.forward(input, None);
        let cell_state = cell_state_batch.select(0, Tensor::arange(0..1)).squeeze(0);
        let hidden_state = hidden_state_batch
            .select(0, Tensor::arange(0..1))
            .squeeze(0);
        cell_state
            .to_data()
            .assert_approx_eq(&Data::from([[0.046]]), 3);
        hidden_state
            .to_data()
            .assert_approx_eq(&Data::from([[0.024]]), 3)
    }
}

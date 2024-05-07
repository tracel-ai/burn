use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::nn::rnn::gate_controller;
use crate::nn::Initializer;
use crate::tensor::activation;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

use super::gate_controller::GateController;

/// Configuration to create a [gru](Gru) module using the [init function](GruConfig::init).
#[derive(Config)]
pub struct GruConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the Gru transformation.
    pub bias: bool,
    /// Gru initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
}

/// The Gru (Gated recurrent unit) module. This implementation is for a unidirectional, stateless, Gru.
///
/// Introduced in the paper: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078).
///
/// Should be created with [GruConfig].
#[derive(Module, Debug)]
pub struct Gru<B: Backend> {
    update_gate: GateController<B>,
    reset_gate: GateController<B>,
    new_gate: GateController<B>,
    d_hidden: usize,
}

impl GruConfig {
    /// Initialize a new [gru](Gru) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gru<B> {
        let d_output = self.d_hidden;

        let update_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
            device,
        );
        let reset_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
            device,
        );
        let new_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
            device,
        );

        Gru {
            update_gate,
            reset_gate,
            new_gate,
            d_hidden: self.d_hidden,
        }
    }
}

impl<B: Backend> Gru<B> {
    /// Applies the forward pass on the input tensor. This GRU implementation
    /// returns a single state tensor with dimensions [batch_size, sequence_length, hidden_size].
    ///
    /// # Shapes
    /// - batched_input: `[batch_size, sequence_length, input_size]`.
    /// - state: An optional tensor representing an initial cell state with the same dimensions
    ///          as batched_input. If none is provided, one will be generated.
    /// - output: `[batch_size, sequence_length, hidden_size]`.
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length, _] = batched_input.shape().dims;

        let mut hidden_state = match state {
            Some(state) => state,
            None => Tensor::zeros(
                [batch_size, seq_length, self.d_hidden],
                &batched_input.device(),
            ),
        };

        for (t, (input_t, hidden_t)) in batched_input
            .iter_dim(1)
            .zip(hidden_state.clone().iter_dim(1))
            .enumerate()
        {
            let input_t = input_t.squeeze(1);
            let hidden_t = hidden_t.squeeze(1);
            // u(pdate)g(ate) tensors
            let biased_ug_input_sum = self.gate_product(&input_t, &hidden_t, &self.update_gate);
            let update_values = activation::sigmoid(biased_ug_input_sum); // Colloquially referred to as z(t)

            // r(eset)g(ate) tensors
            let biased_rg_input_sum = self.gate_product(&input_t, &hidden_t, &self.reset_gate);
            let reset_values = activation::sigmoid(biased_rg_input_sum); // Colloquially referred to as r(t)
            let reset_t = hidden_t.clone().mul(reset_values); // Passed as input to new_gate

            // n(ew)g(ate) tensor
            let biased_ng_input_sum = self.gate_product(&input_t, &reset_t, &self.new_gate);
            let candidate_state = biased_ng_input_sum.tanh(); // Colloquially referred to as g(t)

            // calculate linear interpolation between previous hidden state and candidate state:
            // g(t) * (1 - z(t)) + z(t) * hidden_t
            let state_vector = candidate_state
                .clone()
                .mul(update_values.clone().sub_scalar(1).mul_scalar(-1)) // (1 - z(t)) = -(z(t) - 1)
                + update_values.clone().mul(hidden_t);

            let current_shape = state_vector.shape().dims;
            let unsqueezed_shape = [current_shape[0], 1, current_shape[1]];
            let reshaped_state_vector = state_vector.reshape(unsqueezed_shape);
            hidden_state = hidden_state.slice_assign(
                [0..batch_size, t..(t + 1), 0..self.d_hidden],
                reshaped_state_vector,
            );
        }

        hidden_state
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
    use crate::tensor::{Data, Distribution};
    use crate::{module::Param, nn::LinearRecord, TestBackend};

    /// Test forward pass with simple input vector.
    ///
    /// z_t = sigmoid(0.5*0.1 + 0.5*0) = 0.5125
    /// r_t = sigmoid(0.6*0.1 + 0.*0) = 0.5150
    /// g_t = tanh(0.7*0.1 + 0.7*0) = 0.0699
    ///
    /// h_t = z_t * h' + (1 - z_t) * g_t = 0.0341
    #[test]
    fn tests_forward_single_input_single_feature() {
        TestBackend::seed(0);
        let config = GruConfig::new(1, 1, false);
        let device = Default::default();
        let mut gru = config.init::<TestBackend>(&device);

        fn create_gate_controller(
            weights: f32,
            biases: f32,
            d_input: usize,
            d_output: usize,
            bias: bool,
            initializer: Initializer,
            device: &<TestBackend as Backend>::Device,
        ) -> GateController<TestBackend> {
            let record_1 = LinearRecord {
                weight: Param::from_data(Data::from([[weights]]), device),
                bias: Some(Param::from_data(Data::from([biases]), device)),
            };
            let record_2 = LinearRecord {
                weight: Param::from_data(Data::from([[weights]]), device),
                bias: Some(Param::from_data(Data::from([biases]), device)),
            };
            gate_controller::GateController::create_with_weights(
                d_input,
                d_output,
                bias,
                initializer,
                record_1,
                record_2,
            )
        }

        gru.update_gate = create_gate_controller(
            0.5,
            0.0,
            1,
            1,
            false,
            Initializer::XavierNormal { gain: 1.0 },
            &device,
        );
        gru.reset_gate = create_gate_controller(
            0.6,
            0.0,
            1,
            1,
            false,
            Initializer::XavierNormal { gain: 1.0 },
            &device,
        );
        gru.new_gate = create_gate_controller(
            0.7,
            0.0,
            1,
            1,
            false,
            Initializer::XavierNormal { gain: 1.0 },
            &device,
        );

        let input = Tensor::<TestBackend, 3>::from_data(Data::from([[[0.1]]]), &device);

        let state = gru.forward(input, None);

        let output = state.select(0, Tensor::arange(0..1, &device)).squeeze(0);

        output.to_data().assert_approx_eq(&Data::from([[0.034]]), 3);
    }

    #[test]
    fn test_batched_forward_pass() {
        let device = Default::default();
        let gru = GruConfig::new(64, 1024, true).init::<TestBackend>(&device);
        let batched_input =
            Tensor::<TestBackend, 3>::random([8, 10, 64], Distribution::Default, &device);

        let hidden_state = gru.forward(batched_input, None);

        assert_eq!(hidden_state.shape().dims, [8, 10, 1024]);
    }
}

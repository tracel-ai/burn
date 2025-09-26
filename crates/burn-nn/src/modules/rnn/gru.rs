use burn_core as burn;

use super::gate_controller::GateController;
use burn::config::Config;
use burn::module::Initializer;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;

/// Configuration to create a [gru](Gru) module using the [init function](GruConfig::init).
#[derive(Config, Debug)]
pub struct GruConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the Gru transformation.
    pub bias: bool,
    /// If reset gate should be applied after weight multiplication.
    ///
    /// This configuration option controls how the reset gate is applied to the hidden state.
    /// * `true` - (Default) Match the initial arXiv version of the paper [Learning Phrase Representations using RNN Encoder-Decoder for
    ///   Statistical Machine Translation (v1)](https://arxiv.org/abs/1406.1078v1) and apply the reset gate after multiplication by
    ///   the weights. This matches the behavior of [PyTorch GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU).
    /// * `false` - Match the most recent revision of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine
    ///   Translation (v3)](https://arxiv.org/abs/1406.1078) and apply the reset gate before the weight multiplication.
    ///
    /// The differing implementations can give slightly different numerical results and have different efficiencies. For more
    /// motivation for why the `true` can be more efficient see [Optimizing RNNs with Differentiable Graphs](https://svail.github.io/diff_graphs).
    ///
    /// To set this field to `false` use [`with_reset_after`](`GruConfig::with_reset_after`).
    #[config(default = "true")]
    pub reset_after: bool,
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
#[module(custom_display)]
pub struct Gru<B: Backend> {
    /// The update gate controller.
    pub update_gate: GateController<B>,
    /// The reset gate controller.
    pub reset_gate: GateController<B>,
    /// The new gate controller.
    pub new_gate: GateController<B>,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If reset gate should be applied after weight multiplication.
    pub reset_after: bool,
}

impl<B: Backend> ModuleDisplay for Gru<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, _] = self.update_gate.input_transform.weight.shape().dims();
        let bias = self.update_gate.input_transform.bias.is_some();

        content
            .add("d_input", &d_input)
            .add("d_hidden", &self.d_hidden)
            .add("bias", &bias)
            .add("reset_after", &self.reset_after)
            .optional()
    }
}

impl GruConfig {
    /// Initialize a new [gru](Gru) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gru<B> {
        let d_output = self.d_hidden;

        let update_gate = GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
            device,
        );
        let reset_gate = GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
            device,
        );
        let new_gate = GateController::new(
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
            reset_after: self.reset_after,
        }
    }
}

impl<B: Backend> Gru<B> {
    /// Applies the forward pass on the input tensor. This GRU implementation
    /// returns a state tensor with dimensions `[batch_size, sequence_length, hidden_size]`.
    ///
    /// # Parameters
    /// - batched_input: `[batch_size, sequence_length, input_size]`.
    /// - state: An optional tensor representing an initial cell state with dimensions
    ///   `[batch_size, hidden_size]`. If none is provided, an empty state will be used.
    ///
    /// # Returns
    /// - output: `[batch_size, sequence_length, hidden_size]`
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let device = batched_input.device();
        let [batch_size, seq_length, _] = batched_input.shape().dims();

        let mut batched_hidden_state =
            Tensor::empty([batch_size, seq_length, self.d_hidden], &device);

        let mut hidden_t = match state {
            Some(state) => state,
            None => Tensor::zeros([batch_size, self.d_hidden], &device),
        };

        for (t, input_t) in batched_input.iter_dim(1).enumerate() {
            let input_t = input_t.squeeze_dim(1);
            // u(pdate)g(ate) tensors
            let biased_ug_input_sum =
                self.gate_product(&input_t, &hidden_t, None, &self.update_gate);
            let update_values = activation::sigmoid(biased_ug_input_sum); // Colloquially referred to as z(t)

            // r(eset)g(ate) tensors
            let biased_rg_input_sum =
                self.gate_product(&input_t, &hidden_t, None, &self.reset_gate);
            let reset_values = activation::sigmoid(biased_rg_input_sum); // Colloquially referred to as r(t)

            // n(ew)g(ate) tensor
            let biased_ng_input_sum = if self.reset_after {
                self.gate_product(&input_t, &hidden_t, Some(&reset_values), &self.new_gate)
            } else {
                let reset_t = hidden_t.clone().mul(reset_values); // Passed as input to new_gate
                self.gate_product(&input_t, &reset_t, None, &self.new_gate)
            };
            let candidate_state = biased_ng_input_sum.tanh(); // Colloquially referred to as g(t)

            // calculate linear interpolation between previous hidden state and candidate state:
            // g(t) * (1 - z(t)) + z(t) * hidden_t
            hidden_t = candidate_state
                .clone()
                .mul(update_values.clone().sub_scalar(1).mul_scalar(-1)) // (1 - z(t)) = -(z(t) - 1)
                + update_values.clone().mul(hidden_t);

            let unsqueezed_hidden_state = hidden_t.clone().unsqueeze_dim(1);

            batched_hidden_state = batched_hidden_state.slice_assign(
                [0..batch_size, t..(t + 1), 0..self.d_hidden],
                unsqueezed_hidden_state,
            );
        }

        batched_hidden_state
    }

    /// Helper function for performing weighted matrix product for a gate and adds
    /// bias, if any, and optionally applies reset to hidden state.
    ///
    ///  Mathematically, performs `Wx*X + r .* (Wh*H + b)`, where:
    ///     Wx = weight matrix for the connection to input vector X
    ///     Wh = weight matrix for the connection to hidden state H
    ///     X = input vector
    ///     H = hidden state
    ///     b = bias terms
    ///     r = reset state
    fn gate_product(
        &self,
        input: &Tensor<B, 2>,
        hidden: &Tensor<B, 2>,
        reset: Option<&Tensor<B, 2>>,
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

        match (input_bias, hidden_bias, reset) {
            (Some(input_bias), Some(hidden_bias), Some(r)) => {
                input_product
                    + input_bias.unsqueeze()
                    + r.clone().mul(hidden_product + hidden_bias.unsqueeze())
            }
            (Some(input_bias), Some(hidden_bias), None) => {
                input_product + input_bias.unsqueeze() + hidden_product + hidden_bias.unsqueeze()
            }
            (Some(input_bias), None, Some(r)) => {
                input_product + input_bias.unsqueeze() + r.clone().mul(hidden_product)
            }
            (Some(input_bias), None, None) => {
                input_product + input_bias.unsqueeze() + hidden_product
            }
            (None, Some(hidden_bias), Some(r)) => {
                input_product + r.clone().mul(hidden_product + hidden_bias.unsqueeze())
            }
            (None, Some(hidden_bias), None) => {
                input_product + hidden_product + hidden_bias.unsqueeze()
            }
            (None, None, Some(r)) => input_product + r.clone().mul(hidden_product),
            (None, None, None) => input_product + hidden_product,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearRecord, TestBackend};
    use burn::module::Param;
    use burn::tensor::{Distribution, TensorData};
    use burn::tensor::{Tolerance, ops::FloatElem};

    type FT = FloatElem<TestBackend>;

    fn init_gru<B: Backend>(reset_after: bool, device: &B::Device) -> Gru<B> {
        fn create_gate_controller<B: Backend>(
            weights: f32,
            biases: f32,
            d_input: usize,
            d_output: usize,
            bias: bool,
            initializer: Initializer,
            device: &B::Device,
        ) -> GateController<B> {
            let record_1 = LinearRecord {
                weight: Param::from_data(TensorData::from([[weights]]), device),
                bias: Some(Param::from_data(TensorData::from([biases]), device)),
            };
            let record_2 = LinearRecord {
                weight: Param::from_data(TensorData::from([[weights]]), device),
                bias: Some(Param::from_data(TensorData::from([biases]), device)),
            };
            GateController::create_with_weights(
                d_input,
                d_output,
                bias,
                initializer,
                record_1,
                record_2,
            )
        }

        let config = GruConfig::new(1, 1, false).with_reset_after(reset_after);
        let mut gru = config.init::<B>(device);

        gru.update_gate = create_gate_controller(
            0.5,
            0.0,
            1,
            1,
            false,
            Initializer::XavierNormal { gain: 1.0 },
            device,
        );
        gru.reset_gate = create_gate_controller(
            0.6,
            0.0,
            1,
            1,
            false,
            Initializer::XavierNormal { gain: 1.0 },
            device,
        );
        gru.new_gate = create_gate_controller(
            0.7,
            0.0,
            1,
            1,
            false,
            Initializer::XavierNormal { gain: 1.0 },
            device,
        );
        gru
    }

    /// Test forward pass with simple input vector.
    ///
    /// z_t = sigmoid(0.5*0.1 + 0.5*0) = 0.5125
    /// r_t = sigmoid(0.6*0.1 + 0.*0) = 0.5150
    /// g_t = tanh(0.7*0.1 + 0.7*0) = 0.0699
    ///
    /// h_t = z_t * h' + (1 - z_t) * g_t = 0.0341
    #[test]
    fn tests_forward_single_input_single_feature() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let mut gru = init_gru::<TestBackend>(false, &device);

        let input = Tensor::<TestBackend, 3>::from_data(TensorData::from([[[0.1]]]), &device);
        let expected = TensorData::from([[0.034]]);

        // Reset gate applied to hidden state before the matrix multiplication
        let state = gru.forward(input.clone(), None);

        let output = state
            .select(0, Tensor::arange(0..1, &device))
            .squeeze_dim::<2>(0);

        let tolerance = Tolerance::default();
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        // Reset gate applied to hidden state after the matrix multiplication
        gru.reset_after = true; // override forward behavior
        let state = gru.forward(input, None);

        let output = state
            .select(0, Tensor::arange(0..1, &device))
            .squeeze_dim::<2>(0);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);
    }

    #[test]
    fn tests_forward_seq_len_3() {
        let device = Default::default();
        TestBackend::seed(&device, 0);
        let mut gru = init_gru::<TestBackend>(true, &device);

        let input =
            Tensor::<TestBackend, 3>::from_data(TensorData::from([[[0.1], [0.2], [0.3]]]), &device);
        let expected = TensorData::from([[0.0341], [0.0894], [0.1575]]);

        let result = gru.forward(input.clone(), None);
        let output = result
            .select(0, Tensor::arange(0..1, &device))
            .squeeze_dim::<2>(0);

        let tolerance = Tolerance::default();
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        // Reset gate applied to hidden state before the matrix multiplication
        gru.reset_after = false; // override forward behavior
        let state = gru.forward(input, None);

        let output = state
            .select(0, Tensor::arange(0..1, &device))
            .squeeze_dim::<2>(0);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);
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

    #[test]
    fn display() {
        let config = GruConfig::new(2, 8, true);

        let layer = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{layer}"),
            "Gru {d_input: 2, d_hidden: 8, bias: true, reset_after: true, params: 288}"
        );
    }
}

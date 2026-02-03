use burn_core as burn;

use super::gate_controller::GateController;
use crate::activation::{Activation, ActivationConfig};
use burn::config::Config;
use burn::module::Initializer;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
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
    /// Activation function for the update and reset gates.
    /// Default is Sigmoid, which is standard for GRU gates.
    #[config(default = "ActivationConfig::Sigmoid")]
    pub gate_activation: ActivationConfig,
    /// Activation function for the new/candidate gate.
    /// Default is Tanh, which is standard for GRU.
    #[config(default = "ActivationConfig::Tanh")]
    pub hidden_activation: ActivationConfig,
    /// Optional hidden state clip threshold. If provided, hidden state values are clipped
    /// to the range `[-clip, +clip]` after each timestep. This can help prevent
    /// exploding values during inference.
    pub clip: Option<f64>,
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
    /// Activation function for gates (update, reset).
    pub gate_activation: Activation<B>,
    /// Activation function for new/candidate gate.
    pub hidden_activation: Activation<B>,
    /// Optional hidden state clip threshold.
    pub clip: Option<f64>,
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
            gate_activation: self.gate_activation.init(device),
            hidden_activation: self.hidden_activation.init(device),
            clip: self.clip,
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

        self.forward_iter(
            batched_input.iter_dim(1).zip(0..seq_length),
            state,
            batch_size,
            seq_length,
            &device,
        )
        .0
    }

    /// Forward pass variant that accepts an iterator over timesteps.
    /// Used by BiGru to process sequences in either direction.
    ///
    /// # Parameters
    /// - input_timestep_iter: Iterator yielding (input_tensor, timestep_index) pairs.
    ///   The timestep_index determines where in the output tensor to store results.
    /// - state: Optional initial hidden state with shape `[batch_size, hidden_size]`.
    /// - batch_size: Batch size of the input.
    /// - seq_length: Sequence length of the input.
    /// - device: Device to create tensors on.
    ///
    /// # Returns
    /// - output: `[batch_size, sequence_length, hidden_size]`
    /// - final_hidden: Final hidden state `[batch_size, hidden_size]`
    pub(crate) fn forward_iter<I: Iterator<Item = (Tensor<B, 3>, usize)>>(
        &self,
        input_timestep_iter: I,
        state: Option<Tensor<B, 2>>,
        batch_size: usize,
        seq_length: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let mut batched_hidden_state =
            Tensor::empty([batch_size, seq_length, self.d_hidden], device);

        let mut hidden_t = match state {
            Some(state) => state,
            None => Tensor::zeros([batch_size, self.d_hidden], device),
        };

        for (input_t, t) in input_timestep_iter {
            let input_t = input_t.squeeze_dim(1);

            // u(pdate)g(ate) tensors
            let biased_ug_input_sum =
                self.gate_product(&input_t, &hidden_t, None, &self.update_gate);
            let update_values = self.gate_activation.forward(biased_ug_input_sum);

            // r(eset)g(ate) tensors
            let biased_rg_input_sum =
                self.gate_product(&input_t, &hidden_t, None, &self.reset_gate);
            let reset_values = self.gate_activation.forward(biased_rg_input_sum);

            // n(ew)g(ate) tensor
            let biased_ng_input_sum = if self.reset_after {
                self.gate_product(&input_t, &hidden_t, Some(&reset_values), &self.new_gate)
            } else {
                let reset_t = hidden_t.clone().mul(reset_values);
                self.gate_product(&input_t, &reset_t, None, &self.new_gate)
            };
            let candidate_state = self.hidden_activation.forward(biased_ng_input_sum);

            // calculate linear interpolation between previous hidden state and candidate state:
            // h_t = (1 - z_t) * g_t + z_t * h_{t-1}
            let one_minus_z = update_values.clone().neg().add_scalar(1.0);
            hidden_t = candidate_state.mul(one_minus_z) + update_values.mul(hidden_t);

            // Apply hidden state clipping if configured
            if let Some(clip) = self.clip {
                hidden_t = hidden_t.clamp(-clip, clip);
            }

            let unsqueezed_hidden_state = hidden_t.clone().unsqueeze_dim(1);

            batched_hidden_state = batched_hidden_state.slice_assign(
                [0..batch_size, t..(t + 1), 0..self.d_hidden],
                unsqueezed_hidden_state,
            );
        }

        (batched_hidden_state, hidden_t)
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

        let input_part = match &gate.input_transform.bias {
            Some(bias) => input_product + bias.val().unsqueeze(),
            None => input_product,
        };

        let hidden_part = match &gate.hidden_transform.bias {
            Some(bias) => hidden_product + bias.val().unsqueeze(),
            None => hidden_product,
        };

        match reset {
            Some(r) => input_part + r.clone().mul(hidden_part),
            None => input_part + hidden_part,
        }
    }
}

/// Configuration to create a [BiGru](BiGru) module using the [init function](BiGruConfig::init).
#[derive(Config, Debug)]
pub struct BiGruConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the BiGru transformation.
    pub bias: bool,
    /// If reset gate should be applied after weight multiplication.
    #[config(default = "true")]
    pub reset_after: bool,
    /// BiGru initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
    /// If true, the input tensor is expected to be `[batch_size, seq_length, input_size]`.
    /// If false, the input tensor is expected to be `[seq_length, batch_size, input_size]`.
    #[config(default = true)]
    pub batch_first: bool,
    /// Activation function for the update and reset gates.
    #[config(default = "ActivationConfig::Sigmoid")]
    pub gate_activation: ActivationConfig,
    /// Activation function for the new/candidate gate.
    #[config(default = "ActivationConfig::Tanh")]
    pub hidden_activation: ActivationConfig,
    /// Optional hidden state clip threshold.
    pub clip: Option<f64>,
}

/// The BiGru module. This implementation is for Bidirectional GRU.
///
/// Based on the paper: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078).
///
/// Should be created with [BiGruConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct BiGru<B: Backend> {
    /// GRU for the forward direction.
    pub forward: Gru<B>,
    /// GRU for the reverse direction.
    pub reverse: Gru<B>,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If true, input is `[batch_size, seq_length, input_size]`.
    /// If false, input is `[seq_length, batch_size, input_size]`.
    pub batch_first: bool,
}

impl<B: Backend> ModuleDisplay for BiGru<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, _] = self
            .forward
            .update_gate
            .input_transform
            .weight
            .shape()
            .dims();
        let bias = self.forward.update_gate.input_transform.bias.is_some();

        content
            .add("d_input", &d_input)
            .add("d_hidden", &self.d_hidden)
            .add("bias", &bias)
            .optional()
    }
}

impl BiGruConfig {
    /// Initialize a new [Bidirectional GRU](BiGru) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiGru<B> {
        // Internal GRUs always use batch_first=true; BiGru handles layout conversion
        let base_config = GruConfig::new(self.d_input, self.d_hidden, self.bias)
            .with_initializer(self.initializer.clone())
            .with_reset_after(self.reset_after)
            .with_gate_activation(self.gate_activation.clone())
            .with_hidden_activation(self.hidden_activation.clone())
            .with_clip(self.clip);

        BiGru {
            forward: base_config.clone().init(device),
            reverse: base_config.init(device),
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
        }
    }
}

impl<B: Backend> BiGru<B> {
    /// Applies the forward pass on the input tensor. This Bidirectional GRU implementation
    /// returns the state for each element in a sequence (i.e., across seq_length) and a final state.
    ///
    /// ## Parameters:
    /// - batched_input: The input tensor of shape:
    ///   - `[batch_size, sequence_length, input_size]` if `batch_first` is true (default)
    ///   - `[sequence_length, batch_size, input_size]` if `batch_first` is false
    /// - state: An optional tensor representing the initial hidden state with shape
    ///   `[2, batch_size, hidden_size]`. If no initial state is provided, it is initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor representing the output features. Shape:
    ///   - `[batch_size, sequence_length, hidden_size * 2]` if `batch_first` is true
    ///   - `[sequence_length, batch_size, hidden_size * 2]` if `batch_first` is false
    /// - state: The final forward and reverse hidden states stacked along dimension 0
    ///   with shape `[2, batch_size, hidden_size]`.
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // Convert to batch-first layout internally if needed
        let batched_input = if self.batch_first {
            batched_input
        } else {
            batched_input.swap_dims(0, 1)
        };

        let device = batched_input.clone().device();
        let [batch_size, seq_length, _] = batched_input.shape().dims();

        let [init_state_forward, init_state_reverse] = match state {
            Some(state) => {
                let hidden_state_forward = state
                    .clone()
                    .slice([0..1, 0..batch_size, 0..self.d_hidden])
                    .squeeze_dim(0);
                let hidden_state_reverse = state
                    .slice([1..2, 0..batch_size, 0..self.d_hidden])
                    .squeeze_dim(0);

                [Some(hidden_state_forward), Some(hidden_state_reverse)]
            }
            None => [None, None],
        };

        // forward direction
        let (batched_hidden_state_forward, final_state_forward) = self.forward.forward_iter(
            batched_input.clone().iter_dim(1).zip(0..seq_length),
            init_state_forward,
            batch_size,
            seq_length,
            &device,
        );

        // reverse direction
        let (batched_hidden_state_reverse, final_state_reverse) = self.reverse.forward_iter(
            batched_input.iter_dim(1).rev().zip((0..seq_length).rev()),
            init_state_reverse,
            batch_size,
            seq_length,
            &device,
        );

        let output = Tensor::cat(
            [batched_hidden_state_forward, batched_hidden_state_reverse].to_vec(),
            2,
        );

        // Convert output back to seq-first layout if needed
        let output = if self.batch_first {
            output
        } else {
            output.swap_dims(0, 1)
        };

        let state = Tensor::stack([final_state_forward, final_state_reverse].to_vec(), 0);

        (output, state)
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

    #[test]
    fn test_bigru_batched_forward_pass() {
        let device = Default::default();
        let bigru = BiGruConfig::new(64, 1024, true).init::<TestBackend>(&device);
        let batched_input =
            Tensor::<TestBackend, 3>::random([8, 10, 64], Distribution::Default, &device);

        let (output, state) = bigru.forward(batched_input, None);

        // Output should have hidden_size * 2 features (forward + reverse concatenated)
        assert_eq!(output.shape().dims, [8, 10, 2048]);
        // State should have shape [2, batch_size, hidden_size]
        assert_eq!(state.shape().dims, [2, 8, 1024]);
    }

    #[test]
    fn test_bigru_with_initial_state() {
        let device = Default::default();
        let bigru = BiGruConfig::new(32, 64, true).init::<TestBackend>(&device);
        let batched_input =
            Tensor::<TestBackend, 3>::random([4, 5, 32], Distribution::Default, &device);
        let initial_state =
            Tensor::<TestBackend, 3>::random([2, 4, 64], Distribution::Default, &device);

        let (output, state) = bigru.forward(batched_input, Some(initial_state));

        assert_eq!(output.shape().dims, [4, 5, 128]);
        assert_eq!(state.shape().dims, [2, 4, 64]);
    }

    #[test]
    fn test_bigru_seq_first() {
        let device = Default::default();
        let bigru = BiGruConfig::new(32, 64, true)
            .with_batch_first(false)
            .init::<TestBackend>(&device);
        // Input shape: [seq_length, batch_size, input_size] when batch_first=false
        let batched_input =
            Tensor::<TestBackend, 3>::random([5, 4, 32], Distribution::Default, &device);

        let (output, state) = bigru.forward(batched_input, None);

        // Output shape: [seq_length, batch_size, hidden_size * 2]
        assert_eq!(output.shape().dims, [5, 4, 128]);
        assert_eq!(state.shape().dims, [2, 4, 64]);
    }

    /// Test BiGru against PyTorch reference implementation.
    /// Expected values computed with PyTorch nn.GRU(bidirectional=True).
    #[test]
    fn test_bigru_against_pytorch() {
        use burn::tensor::Device;

        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = BiGruConfig::new(2, 3, true);
        let mut bigru = config.init::<TestBackend>(&device);

        fn create_gate_controller<const D1: usize, const D2: usize>(
            input_weights: [[f32; D1]; D2],
            input_biases: [f32; D1],
            hidden_weights: [[f32; D1]; D1],
            hidden_biases: [f32; D1],
            device: &Device<TestBackend>,
        ) -> GateController<TestBackend> {
            let d_input = input_weights[0].len();
            let d_output = input_weights.len();

            let input_record = LinearRecord {
                weight: Param::from_data(TensorData::from(input_weights), device),
                bias: Some(Param::from_data(TensorData::from(input_biases), device)),
            };
            let hidden_record = LinearRecord {
                weight: Param::from_data(TensorData::from(hidden_weights), device),
                bias: Some(Param::from_data(TensorData::from(hidden_biases), device)),
            };
            GateController::create_with_weights(
                d_input,
                d_output,
                true,
                Initializer::XavierUniform { gain: 1.0 },
                input_record,
                hidden_record,
            )
        }

        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[
                [0.949, -0.861],
                [0.892, 0.927],
                [-0.173, -0.301],
                [-0.081, 0.992],
            ]]),
            &device,
        );
        let h0 = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[0.280, 0.360, -1.242]], [[-0.588, 0.729, -0.788]]]),
            &device,
        );

        // Forward GRU gates (weights from PyTorch with seed 42, transposed for burn)
        bigru.forward.update_gate = create_gate_controller(
            [[-0.2811, 0.5090, 0.5018], [0.3391, -0.4236, 0.1081]],
            [0.2932, -0.3519, -0.5715],
            [
                [-0.3471, 0.5214, 0.0961],
                [0.0545, -0.4904, -0.1875],
                [-0.5702, 0.4457, 0.3568],
            ],
            [-0.0100, 0.4518, -0.4102],
            &device,
        );

        bigru.forward.reset_gate = create_gate_controller(
            [[0.4414, -0.1353, -0.1265], [0.4792, 0.5304, 0.1165]],
            [-0.2524, 0.3333, 0.1033],
            [
                [-0.2695, -0.0677, -0.4557],
                [0.1472, -0.2345, -0.2662],
                [-0.2660, 0.3830, -0.1630],
            ],
            [0.1663, 0.2391, 0.1826],
            &device,
        );

        bigru.forward.new_gate = create_gate_controller(
            [[0.4266, 0.2784, 0.4451], [0.0782, -0.0815, 0.0853]],
            [-0.2231, -0.4428, 0.4737],
            [
                [0.0900, -0.1821, 0.2430],
                [0.4665, 0.1551, 0.5155],
                [0.0631, -0.1566, 0.3337],
            ],
            [0.0364, -0.3941, 0.1780],
            &device,
        );

        // Reverse GRU gates
        bigru.reverse.update_gate = create_gate_controller(
            [[-0.3444, 0.1924, -0.4765], [0.5193, 0.5556, -0.5727]],
            [0.1090, 0.1779, -0.5385],
            [
                [0.1221, 0.3925, 0.5287],
                [-0.1472, -0.4187, -0.1948],
                [0.3441, -0.3082, -0.2047],
            ],
            [0.0016, -0.2148, -0.0400],
            &device,
        );

        bigru.reverse.reset_gate = create_gate_controller(
            [[-0.1988, -0.1203, -0.3422], [0.1769, 0.4788, -0.3443]],
            [-0.5053, -0.3676, 0.5771],
            [
                [-0.3936, 0.3504, -0.4486],
                [0.3063, -0.1370, -0.2914],
                [-0.2334, 0.3303, 0.1760],
            ],
            [-0.5080, -0.2488, -0.3456],
            &device,
        );

        bigru.reverse.new_gate = create_gate_controller(
            [[-0.4517, 0.2339, 0.4797], [-0.3884, 0.2067, -0.2982]],
            [-0.3792, -0.1922, 0.0903],
            [
                [-0.5586, -0.0762, -0.3944],
                [-0.3306, -0.4191, -0.4898],
                [0.1442, 0.0135, -0.3179],
            ],
            [-0.3912, -0.3963, -0.3368],
            &device,
        );

        // Expected values from PyTorch
        let expected_output_with_init = TensorData::from([[
            [0.24537, 0.14018, 0.19449, -0.49777, -0.15647, 0.48392],
            [0.27468, -0.14514, 0.56205, -0.60381, -0.04986, 0.15683],
            [-0.04062, -0.33486, 0.52330, -0.42244, -0.12644, -0.12034],
            [-0.11743, -0.53873, 0.54429, -0.64943, 0.30127, -0.41943],
        ]]);

        let expected_hn_with_init = TensorData::from([
            [[-0.11743, -0.53873, 0.54429]],
            [[-0.49777, -0.15647, 0.48392]],
        ]);

        let expected_output_without_init = TensorData::from([[
            [0.07452, -0.08247, 0.46677, -0.46770, -0.18086, 0.47519],
            [0.15843, -0.27144, 0.65781, -0.50286, -0.12806, 0.14884],
            [-0.10704, -0.41573, 0.53954, -0.24794, -0.24003, -0.10294],
            [-0.16505, -0.57952, 0.53565, -0.23598, -0.07137, -0.28937],
        ]]);

        let expected_hn_without_init = TensorData::from([
            [[-0.16505, -0.57952, 0.53565]],
            [[-0.46770, -0.18086, 0.47519]],
        ]);

        let (output_with_init, hn_with_init) = bigru.forward(input.clone(), Some(h0));
        let (output_without_init, hn_without_init) = bigru.forward(input, None);

        let tolerance = Tolerance::permissive();
        output_with_init
            .to_data()
            .assert_approx_eq::<FT>(&expected_output_with_init, tolerance);
        output_without_init
            .to_data()
            .assert_approx_eq::<FT>(&expected_output_without_init, tolerance);
        hn_with_init
            .to_data()
            .assert_approx_eq::<FT>(&expected_hn_with_init, tolerance);
        hn_without_init
            .to_data()
            .assert_approx_eq::<FT>(&expected_hn_without_init, tolerance);
    }

    #[test]
    fn bigru_display() {
        let config = BiGruConfig::new(2, 8, true);

        let layer = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{layer}"),
            "BiGru {d_input: 2, d_hidden: 8, bias: true, params: 576}"
        );
    }

    #[test]
    fn test_gru_custom_activations() {
        let device = Default::default();

        // Create GRU with custom activations (ReLU instead of Sigmoid/Tanh)
        let config = GruConfig::new(4, 8, true)
            .with_gate_activation(ActivationConfig::Relu)
            .with_hidden_activation(ActivationConfig::Relu);
        let gru = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([2, 3, 4], Distribution::Default, &device);

        // Should run without panicking and produce valid output
        let output = gru.forward(input, None);
        assert_eq!(output.shape().dims, [2, 3, 8]);
    }

    #[test]
    fn test_bigru_custom_activations() {
        let device = Default::default();

        // Create BiGRU with custom activations
        let config = BiGruConfig::new(4, 8, true)
            .with_gate_activation(ActivationConfig::Relu)
            .with_hidden_activation(ActivationConfig::Relu);
        let bigru = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([2, 3, 4], Distribution::Default, &device);

        let (output, state) = bigru.forward(input, None);
        assert_eq!(output.shape().dims, [2, 3, 16]); // hidden_size * 2
        assert_eq!(state.shape().dims, [2, 2, 8]);
    }

    #[test]
    fn test_gru_clipping() {
        let device = Default::default();

        // Create GRU with clipping enabled
        let clip_value = 0.5;
        let config = GruConfig::new(4, 8, true).with_clip(Some(clip_value));
        let gru = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([2, 5, 4], Distribution::Default, &device);

        let output = gru.forward(input, None);

        // Verify output values are within the clip range
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();
        for val in output_data {
            assert!(
                val >= -clip_value as f32 && val <= clip_value as f32,
                "Value {} is outside clip range [-{}, {}]",
                val,
                clip_value,
                clip_value
            );
        }
    }

    #[test]
    fn test_bigru_clipping() {
        let device = Default::default();

        // Create BiGRU with clipping enabled
        let clip_value = 0.3;
        let config = BiGruConfig::new(4, 8, true).with_clip(Some(clip_value));
        let bigru = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([2, 5, 4], Distribution::Default, &device);

        let (output, state) = bigru.forward(input, None);

        // Verify output values are within the clip range
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();
        for val in output_data {
            assert!(
                val >= -clip_value as f32 && val <= clip_value as f32,
                "Output value {} is outside clip range [-{}, {}]",
                val,
                clip_value,
                clip_value
            );
        }

        // Verify state values are within the clip range
        let state_data: Vec<f32> = state.to_data().to_vec().unwrap();
        for val in state_data {
            assert!(
                val >= -clip_value as f32 && val <= clip_value as f32,
                "State value {} is outside clip range [-{}, {}]",
                val,
                clip_value,
                clip_value
            );
        }
    }
}

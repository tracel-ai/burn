use burn_core as burn;

use crate::activation::{Activation, ActivationConfig};
use crate::{GateController, LstmState, OptionalInitialLstmState};
use alloc::boxed::Box;
use burn::Tensor;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay};
use burn::prelude::Device;
use burn::prelude::s;

/// Configuration to create a [Lstm](Lstm) module using the [init function](LstmConfig::init).
#[derive(Config, Debug)]
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
    /// If true, the input tensor is expected to be `[batch_size, seq_length, input_size]`.
    /// If false, the input tensor is expected to be `[seq_length, batch_size, input_size]`.
    #[config(default = true)]
    pub batch_first: bool,
    /// If true, process the sequence in reverse order.
    /// This is useful for implementing reverse-direction LSTMs (e.g., ONNX reverse direction).
    #[config(default = false)]
    pub reverse: bool,
    /// Optional cell state clip threshold. If provided, cell state values are clipped
    /// to the range `[-clip, +clip]` after each timestep. This can help prevent
    /// exploding values during inference.
    pub clip: Option<f64>,
    /// If true, couples the input and forget gates: `f_t = 1 - i_t`.
    /// This reduces the number of parameters and is based on GRU-style simplification.
    #[config(default = false)]
    pub input_forget: bool,
    /// Activation function for the input, forget, and output gates.
    /// Default is Sigmoid, which is standard for LSTM gates.
    #[config(default = "ActivationConfig::Sigmoid")]
    pub gate_activation: ActivationConfig,
    /// Activation function for the cell gate (candidate cell state).
    /// Default is Tanh, which is standard for LSTM.
    #[config(default = "ActivationConfig::Tanh")]
    pub cell_activation: ActivationConfig,
    /// Activation function applied to the cell state before computing hidden output.
    /// Default is Tanh, which is standard for LSTM.
    #[config(default = "ActivationConfig::Tanh")]
    pub hidden_activation: ActivationConfig,
}

/// The Lstm module. This implementation is for a unidirectional, stateless, Lstm.
///
/// Introduced in the paper: [Long Short-Term Memory](https://www.researchgate.net/publication/13853244).
///
/// Should be created with [LstmConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Lstm {
    /// The input gate regulates which information to update and store in the cell state at each time step.
    pub input_gate: GateController,
    /// The forget gate is used to control which information to discard or keep in the memory cell at each time step.
    /// Note: When `input_forget` is true, this gate is not used (forget = 1 - input).
    pub forget_gate: GateController,
    /// The output gate determines which information from the cell state to output at each time step.
    pub output_gate: GateController,
    /// The cell gate is used to compute the cell state that stores and carries information through time.
    pub cell_gate: GateController,
    /// The hidden state of the LSTM.
    pub d_hidden: usize,
    /// If true, input is `[batch_size, seq_length, input_size]`.
    /// If false, input is `[seq_length, batch_size, input_size]`.
    pub batch_first: bool,
    /// If true, process the sequence in reverse order.
    pub reverse: bool,
    /// Optional cell state clip threshold.
    pub clip: Option<f64>,
    /// If true, couples input and forget gates: f_t = 1 - i_t.
    pub input_forget: bool,
    /// Activation function for gates (input, forget, output).
    pub gate_activation: Activation,
    /// Activation function for cell gate (candidate cell state).
    pub cell_activation: Activation,
    /// Activation function for hidden output.
    pub hidden_activation: Activation,
}

impl ModuleDisplay for Lstm {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, _] = self.input_gate.input_transform.weight.shape().dims();
        let bias = self.input_gate.input_transform.bias.is_some();

        content
            .add("d_input", &d_input)
            .add("d_hidden", &self.d_hidden)
            .add("bias", &bias)
            .optional()
    }
}

impl LstmConfig {
    /// Initialize a new [lstm](Lstm) module.
    pub fn init(&self, device: &Device) -> Lstm {
        let d_output = self.d_hidden;

        let new_gate = || {
            GateController::new(
                self.d_input,
                d_output,
                self.bias,
                self.initializer.clone(),
                device,
            )
        };

        Lstm {
            input_gate: new_gate(),
            forget_gate: new_gate(),
            output_gate: new_gate(),
            cell_gate: new_gate(),
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
            reverse: self.reverse,
            clip: self.clip,
            input_forget: self.input_forget,
            gate_activation: self.gate_activation.init(device),
            cell_activation: self.cell_activation.init(device),
            hidden_activation: self.hidden_activation.init(device),
        }
    }
}

impl Lstm {
    /// Applies the forward pass on the input tensor. This LSTM implementation
    /// returns the state for each element in a sequence (i.e., across seq_length) and a final state.
    ///
    /// ## Parameters:
    /// - batched_input: The input tensor of shape:
    ///   - `[batch_size, sequence_length, input_size]` if `batch_first` is true (default)
    ///   - `[sequence_length, batch_size, input_size]` if `batch_first` is false
    /// - state: An optional `LstmState` representing the initial cell state and hidden state.
    ///   Each state tensor has shape `[batch_size, hidden_size]`.
    ///   If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor represents the output features of LSTM. Shape:
    ///   - `[batch_size, sequence_length, hidden_size]` if `batch_first` is true
    ///   - `[sequence_length, batch_size, hidden_size]` if `batch_first` is false
    /// - state: A `LstmState` represents the final states. Both `state.cell` and `state.hidden` have the shape
    ///   `[batch_size, hidden_size]`.
    pub fn forward(
        &self,
        batched_input: Tensor<3>,
        state: Option<LstmState<2>>,
    ) -> (Tensor<3>, LstmState<2>) {
        // Convert to batch-first layout internally if needed
        let batched_input = if self.batch_first {
            batched_input
        } else {
            batched_input.swap_dims(0, 1)
        };

        let device = batched_input.device();
        let [batch_size, seq_length, _] = batched_input.dims();

        // Process sequence in forward or reverse order based on config
        let it = batched_input.iter_dim(1).enumerate();
        let it: Box<dyn Iterator<Item = (usize, Tensor<3>)>> = if self.reverse {
            Box::new(it.rev())
        } else {
            Box::new(it)
        };

        let (output, state) = self.forward_iter(it, state, batch_size, seq_length, &device);

        // Convert output back to seq-first layout if needed
        let output = if self.batch_first {
            output
        } else {
            output.swap_dims(0, 1)
        };

        (output, state)
    }

    /// Applies the forward iteration over input timesteps for the LSTM. This method is called
    /// internally by the [`forward`](#method.forward) function to process each timestep of the sequence.
    ///
    /// ## Parameters:
    /// - `input_timestep_iter`: An iterator where each item is a pair consisting of the timestep index (`usize`)
    ///   and the corresponding input tensor of shape `[batch_size, 1, d_input]`.
    /// - `state`: An optional [`LstmState`] representing the initial cell and hidden states.
    ///   If `None`, the states are initialized to zeros with shapes `[batch_size, d_hidden]`.
    /// - `batch_size`: The number of sequences in the batch.
    /// - `seq_length`: The length of the input sequence (number of timesteps).
    /// - `device`: The device where computations will run.
    ///
    /// ## Returns:
    /// - A pair where:
    ///   - The first element is a tensor of shape `[batch_size, seq_length, d_hidden]`,
    ///     containing the output features for all timesteps.
    ///   - The second element is the final [`LstmState`], which includes the final cell and hidden states,
    ///     both with the shape `[batch_size, d_hidden]`.
    pub fn forward_iter<I: Iterator<Item = (usize, Tensor<3>)>>(
        &self,
        input_timestep_iter: I,
        state: Option<LstmState<2>>,
        batch_size: usize,
        seq_length: usize,
        device: &Device,
    ) -> (Tensor<3>, LstmState<2>) {
        let mut batched_hidden_state =
            Tensor::empty([batch_size, seq_length, self.d_hidden], device);

        let (mut cell_state, mut hidden_state) = state
            .unwrap_or_initial([batch_size, self.d_hidden], device)
            .unpack();

        for (t, input_t) in input_timestep_iter {
            let input_t = input_t.squeeze_dim(1);

            // i(nput)g(ate) tensors
            let biased_ig_input_sum = self
                .input_gate
                .gate_product(input_t.clone(), hidden_state.clone());
            let input_values = self.gate_activation.forward(biased_ig_input_sum);

            // f(orget)g(ate) tensors - either computed or coupled to input gate
            let forget_values = if self.input_forget {
                // Coupled mode: f_t = 1 - i_t
                input_values.clone().neg().add_scalar(1.0)
            } else {
                let biased_fg_input_sum = self
                    .forget_gate
                    .gate_product(input_t.clone(), hidden_state.clone());
                self.gate_activation.forward(biased_fg_input_sum)
            };

            // o(output)g(ate) tensors
            let biased_og_input_sum = self
                .output_gate
                .gate_product(input_t.clone(), hidden_state.clone());
            let output_values = self.gate_activation.forward(biased_og_input_sum);

            // c(ell)g(ate) tensors
            let biased_cg_input_sum = self.cell_gate.gate_product(input_t, hidden_state.clone());
            let candidate_cell_values = self.cell_activation.forward(biased_cg_input_sum);

            cell_state = forget_values * cell_state + input_values * candidate_cell_values;

            // Apply cell state clipping if configured
            if let Some(clip) = self.clip {
                cell_state = cell_state.clamp(-clip, clip);
            }

            hidden_state = output_values * self.hidden_activation.forward(cell_state.clone());

            // store the hidden state for this timestep
            batched_hidden_state = batched_hidden_state
                .slice_assign(s![.., t, ..], hidden_state.clone().unsqueeze_dim(1));
        }

        (
            batched_hidden_state,
            LstmState::new(cell_state, hidden_state),
        )
    }
}

#[cfg(test)]
mod test {
    use crate::{GateController, Linear, LstmConfig};
    use burn_core::Tensor;
    use burn_core::module::{Initializer, Param};
    use burn_core::prelude::Device;
    use burn_core::tensor::{Distribution, ElementConversion, Shape, TensorData, Tolerance};
    pub type FT = f32;

    #[test]
    fn display_lstm() {
        let config = LstmConfig::new(2, 3, true);

        let layer = config.init(&Default::default());

        assert_eq!(
            alloc::format!("{layer}"),
            "Lstm {d_input: 2, d_hidden: 3, bias: true, params: 84}"
        );
    }

    #[test]
    fn test_with_uniform_initializer() {
        let device = Device::default();
        device.seed(0);

        let config = LstmConfig::new(5, 5, false)
            .with_initializer(Initializer::Uniform { min: 0.0, max: 1.0 });
        let lstm = config.init(&Default::default());

        let gate_to_data = |gate: GateController| gate.input_transform.weight.val().to_data();

        gate_to_data(lstm.input_gate).assert_within_range::<FT>(0.elem()..1.elem());
        gate_to_data(lstm.forget_gate).assert_within_range::<FT>(0.elem()..1.elem());
        gate_to_data(lstm.output_gate).assert_within_range::<FT>(0.elem()..1.elem());
        gate_to_data(lstm.cell_gate).assert_within_range::<FT>(0.elem()..1.elem());
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
        let device = Device::default();
        device.seed(0);

        let config = LstmConfig::new(1, 1, false);
        let mut lstm = config.init(&device);

        fn create_gate_controller(weights: f32, biases: f32, device: &Device) -> GateController {
            let record_1 = Linear {
                weight: Param::from_data(TensorData::from([[weights]]), device),
                bias: Some(Param::from_data(TensorData::from([biases]), device)),
            };
            let record_2 = Linear {
                weight: Param::from_data(TensorData::from([[weights]]), device),
                bias: Some(Param::from_data(TensorData::from([biases]), device)),
            };
            GateController::create_with_weights(record_1, record_2)
        }

        lstm.input_gate = create_gate_controller(0.5, 0.0, &device);
        lstm.forget_gate = create_gate_controller(0.7, 0.0, &device);
        lstm.cell_gate = create_gate_controller(0.9, 0.0, &device);
        lstm.output_gate = create_gate_controller(1.1, 0.0, &device);

        // single timestep with single feature
        let input = Tensor::<3>::from_data(TensorData::from([[[0.1]]]), &device);

        let (output, state) = lstm.forward(input, None);

        let expected = TensorData::from([[0.046]]);
        let tolerance = Tolerance::default();
        state
            .cell
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        let expected = TensorData::from([[0.0242]]);
        state
            .hidden
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        output
            .select(0, Tensor::arange(0..1, &device))
            .squeeze_dim::<2>(0)
            .to_data()
            .assert_approx_eq::<FT>(&state.hidden.to_data(), tolerance);
    }

    #[test]
    fn test_batched_forward_pass() {
        let device = Default::default();
        let lstm = LstmConfig::new(64, 1024, true).init(&device);
        let batched_input = Tensor::<3>::random([8, 10, 64], Distribution::Default, &device);

        let (output, state) = lstm.forward(batched_input, None);

        assert_eq!(output.dims(), [8, 10, 1024]);
        assert_eq!(state.cell.dims(), [8, 1024]);
        assert_eq!(state.hidden.dims(), [8, 1024]);
    }

    #[test]
    fn test_batched_forward_pass_batch_of_one() {
        let device = Default::default();
        let lstm = LstmConfig::new(64, 1024, true).init(&device);
        let batched_input = Tensor::<3>::random([1, 2, 64], Distribution::Default, &device);

        let (output, state) = lstm.forward(batched_input, None);

        assert_eq!(output.dims(), [1, 2, 1024]);
        assert_eq!(state.cell.dims(), [1, 1024]);
        assert_eq!(state.hidden.dims(), [1, 1024]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_batched_backward_pass() {
        let device = Device::default().autodiff();
        let lstm = LstmConfig::new(64, 32, true).init(&device);
        let shape: Shape = [8, 10, 64].into();
        let batched_input = Tensor::<3>::random(shape, Distribution::Default, &device);

        let (output, _) = lstm.forward(batched_input.clone(), None);
        let fake_loss = output;
        let grads = fake_loss.backward();

        let some_gradient = lstm
            .output_gate
            .hidden_transform
            .weight
            .grad(&grads)
            .unwrap();

        // Asserts that the gradients exist and are non-zero
        assert_ne!(
            some_gradient
                .any()
                .into_data()
                .iter::<f32>()
                .next()
                .unwrap(),
            0.0
        );
    }
}
